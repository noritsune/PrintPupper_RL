"""PrintPupper 速度コマンド追従歩行タスク (MJX / brax PPO 用)。

設計上の最重要制約:
  PrintPupper の PWM サーボには位置フィードバックが無い。よって一般的な四足 RL が
  観測に入れる「実測関節角・関節速度」は実機では取得できない。本環境は最初から
  その制約で学習する:
    - actor 観測 = IMU(ジャイロ/重力方向) + コマンド + 自分が出した指令の履歴 + 歩容位相
    - critic 観測 = 上記 + 特権情報(実関節角/速度/接触/胴体速度)  [非対称 actor-critic]
  さらに膝は4節リンク駆動なので、MJCF の矩形リミット内でも実機で出せない姿勢がある。
  env.linkage.infeasibility でそれを罰する。
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import gait as pg_gait
from mujoco_playground._src import mjx_env
from mujoco.mjx._src import math as pg_math

from . import constants as consts
from .linkage import infeasibility

# 観測履歴 1 フレームの次元。[gyro 3, accel*scale 3, gravity 3, action 12]。
# 実機で取れる量だけで構成する (関節エンコーダも接地センサも無い)。
# 加速度計を落とさないのが肝心で、接地の瞬間の衝撃は加速度にしか出ない。
# 速度コマンドと歩容位相は履歴に入れない (前者はユーザ入力、後者は内部
# クロックの決定論的な関数なので、過去の値から新しい情報が増えない)。
HIST_FRAME_DIM = 21


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,          # 50 Hz 制御 (実機サーボ PWM は 100 Hz)
        sim_dt=0.004,
        episode_length=1000,   # 20 s
        action_repeat=1,
        action_scale=0.25,     # 基準姿勢からの残差 [rad]
        soft_joint_pos_limit_factor=0.95,
        # --- サーボ (DS3235) ---
        # 値は model/pupper.xml が持つ (scripts/build_mjcf.py が生成)。
        # ここで None 以外を入れた場合のみ上書きする。二重定義は事故のもと。
        Kp=None,
        Kd=None,
        # --- 実機の通信・アクチュエーション遅延 ---
        action_delay=config_dict.create(enable=True, min_steps=1, max_steps=3),
        # --- 観測履歴スタック (actor の記憶) ---
        # actor は IMU と行動履歴しか持たない盲目方策で、既定の記憶は
        # last_act + last_last_act の 2 ステップ = 40ms しかない。遊脚は
        # gait_freq 0.7-1.2Hz で 333-571ms なので、**1 歩の 7-12% しか
        # 見えていない**。「さっき踏んだ足は床か段の上か」「胴体はもう段の
        # 縁を越えたか」は原理的に表現できず、位相クロックという open-loop の
        # 時計しか頼るものがない。
        #
        # 1 フレーム = [gyro 3, accel*scale 3, gravity 3, action 12] = 21 次元を
        # length 個、stride ステップおきに現在の観測の後ろに連結する。
        # 実効の時間窓は length*stride*ctrl_dt 秒。既定の 10*4*0.02 = 0.8s は
        # gait_freq 0.7-1.2Hz の 1 周期 (833-1429ms) にほぼ届く長さ。
        # stride で間引くのは、生の 40 フレームを全部入れると 840 次元になり
        # 隣接フレームがほぼ同一値で冗長なため (Lee et al. 2020 / Margolis 2022
        # の TCN・履歴 MLP と同じ考え方)。
        #
        # length=0 で完全に無効 (観測 38 次元のまま)。**既定は 0**。
        # 既存の重み (u_stand 等) との互換を壊さないため、有効化は
        # train.py --obs-history で明示的に指定したときだけにする。
        # 38 次元の重みを履歴つきに広げるには scripts/adapt_ckpt.py を使う。
        obs_history=config_dict.create(length=0, stride=4),
        # --- IMU モデル (6軸: ジャイロ + 加速度計。MPU6050/ICM-20948 相当) ---
        # 実機のセンサはこれだけ。関節エンコーダも足裏接地センサも無い。
        # bias はエピソード内で一定にする。iid ノイズだけだと方策が平均化で
        # 簡単に打ち消せてしまい、実機の定常オフセットに対する頑健性が付かない。
        imu_config=config_dict.create(
            level=1.0,                  # 0.0 で全ノイズ無効
            gyro_noise=0.05,            # rad/s, 白色
            gyro_bias=0.05,             # rad/s, エピソード内一定
            accel_noise=0.30,           # m/s^2, 白色
            accel_bias=0.30,            # m/s^2, エピソード内一定
            accel_obs_scale=0.1,        # 観測ベクトルの条件数を整えるだけ
            gravity_noise=0.02,         # 姿勢推定の白色誤差 (単位ベクトル)
            gravity_tilt_bias=0.03,     # rad, 取付け角ずれ + 定常推定誤差
            # 相補フィルタが機体加速度をどれだけ姿勢に誤混入させるか。
            # 0 = 完全な姿勢推定 (非現実的), 1 = 加速度計のみのチルト推定。
            gravity_accel_coupling=0.15,
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # 追従
                tracking_lin_vel=2.0,
                tracking_ang_vel=1.0,
                # 胴体
                lin_vel_z=-1.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                base_height=-2.0,
                # 足: 位相クロックが指定する足先高さプロファイルに追従させる。
                # これで歩容周波数と接地デューティが陽に決まり、方策が勝手に
                # 高周波の「跳ねるトロット」を選ぶのを防ぐ。
                feet_phase=1.5,
                # 平地では 0。段差版 (env/terrain.py) だけが有効にする
                feet_clearance=0.0,
                # 段差を「登れたこと」を直接評価する項。平地では 0。
                # 段差版 (env/terrain.py) だけが有効にする。
                climb=0.0,
                feet_slip=-0.15,
                flight=-0.5,          # 4脚とも非接地の時間
                # 正則化
                torques=-0.002,
                # サーボのストールトルクに張り付くのを直接罰する。
                # DS3235 は 3Nm 出せるが、常用でストールさせるとギヤが持たず
                # 熱でも落ちる。0.5*stall を超えたぶんを二次で罰する。
                torque_limit=-1.0,
                action_rate=-0.02,
                energy=-0.001,
                pose=-0.3,
                dof_pos_limits=-2.0,
                stand_still=-0.5,
                termination=-1.0,
                # PrintPupper 固有: 4節リンクで実現不能な姿勢
                linkage_infeasible=-5.0,
            ),
            # sigma はコマンドの分散に対して決める。
            # 追従報酬 exp(-err/sigma) で「静止したまま」の場合 err = E[cmd^2] なので、
            # sigma が大きいと静止でも高得点になり、罰則に負けて歩かなくなる。
            # コマンド分布 (一様 +-a を確率 b で採用) の分散:
            #   lin: var(vx)+var(vy) = 0.30^2/3*0.9 + 0.15^2/3*0.25 = 0.0289
            #   ang: var(wz)         = 1.00^2/3*0.5                 = 0.167
            # 静止時の追従報酬を最大の 1/10 に抑えるには sigma = var/ln(10)。
            tracking_sigma=0.0126,        # 線速度用
            tracking_sigma_ang=0.073,     # 角速度用
            base_height_target=consts.STAND_HEIGHT,
            max_foot_height=consts.SWING_HEIGHT,
            feet_phase_sigma=0.01,
            # 軟限界: この割合を超えたトルクを罰する (DS3235 stall の何割か)
            torque_soft_limit_frac=0.5,
        ),
        pert_config=config_dict.create(
            enable=True,
            velocity_kick=[0.0, 0.8],      # 1.48kg の機体。Go1(12kg) より小さく
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            # gait.yaml の max_linear_velocity_x/y, max_angular_velocity_z
            a=[consts.CMD_VX[1], consts.CMD_VY[1], consts.CMD_WZ[1]],
            b=[0.9, 0.25, 0.5],            # コマンドをゼロにしない確率
        ),
        gait_freq=[consts.GAIT_FREQ[0], consts.GAIT_FREQ[1]],
        impl="warp",
        # naconmax はバッチ全体の接触バッファ。環境あたりの接触点数 x num_envs で
        # 与える。このモデルの接触は「床 vs 足球4 + 胴体箱」で、歩行中は足4点が
        # 主。小さすぎると接触が落ちて機体が床を抜け reward が NaN になるので、
        # scripts/train.py が --num-envs から naconmax = 8 * num_envs を設定する。
        naconmax=8 * 4096,
        njmax=80,
    )


class PupperJoystick(mjx_env.MjxEnv):
    """平地での速度コマンド追従歩行。"""

    # 段差版 (env/terrain.py) が差し替える。
    XML = consts.XML

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        xml = self.XML.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(xml)
        self._mj_model.opt.timestep = self._config.sim_dt
        if self._config.Kd is not None:
            self._mj_model.dof_damping[6:] = self._config.Kd
        if self._config.Kp is not None:
            self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
            self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._edit_model(self._mj_model)          # 段差版が hfield を書き込む
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml
        self._post_init()

    def _post_init(self) -> None:
        m = self._mj_model
        self._init_q = jp.array(m.keyframe("stand").qpos)
        self._default_pose = jp.array(m.keyframe("stand").qpos[7:])
        self._lowers, self._uppers = m.jnt_range[1:].T
        f = self._config.soft_joint_pos_limit_factor
        mid = (self._lowers + self._uppers) / 2
        rng = (self._uppers - self._lowers) / 2
        self._soft_lowers = jp.array(mid - rng * f)
        self._soft_uppers = jp.array(mid + rng * f)

        self._torso_body_id = m.body(consts.ROOT_BODY).id
        self._torso_mass = m.body_subtreemass[self._torso_body_id]
        self._feet_site_id = np.array([m.site(s).id for s in consts.FEET_SITES])
        self._feet_geom_id = np.array([m.geom(g).id for g in consts.FEET_GEOMS])
        self._floor_geom_id = m.geom("floor").id
        self._feet_contact_adr = np.array(
            [m.sensor_adr[m.sensor(s).id] for s in consts.FEET_CONTACT_SENSOR]
        )
        self._trunk_contact_adr = m.sensor_adr[
            m.sensor(consts.TRUNK_CONTACT_SENSOR).id
        ]
        self._feet_linvel_adr = np.array(
            [m.sensor_adr[m.sensor(s).id] for s in consts.FEET_LINVEL_SENSOR]
        )
        self._hip_idx = np.array([0, 3, 6, 9])      # 各脚の hip(外転) 関節
        # トロット: 対角の脚が同相 (lf+rh / rf+lh)
        self._foot_phase_offset = jp.array([0.0, jp.pi, jp.pi, 0.0])
        self._stall = float(self._mj_model.actuator_forcerange[0, 1])
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)
        self._max_delay = int(self._config.action_delay.max_steps)
        # --- 観測履歴スタック ---
        oh = self._config.obs_history
        self._hist_len = int(oh.length)
        self._hist_stride = max(1, int(oh.stride))
        # リングバッファの長さ。obs に載るのは stride おきの length 個だけだが、
        # バッファ自体は毎ステップ詰めるので length*stride 必要。
        self._hist_frames = self._hist_len * self._hist_stride
        # obs に載せるバッファ内の位置。buf[0] が 1 ステップ前。
        self._hist_idx = np.arange(1, self._hist_len + 1) * self._hist_stride - 1

    # ---------- 地形フック ----------
    # 平地版はすべて素通り/ゼロ。段差版 (env/terrain.py) が上書きする。
    # 高さ関係の報酬をここ経由にしておくと、ワールド z 絶対値ではなく
    # 「足元の地面からの相対高さ」になり、段差地形でもそのまま意味を持つ。
    # 平地では _ground_height が厳密に 0 を返すので数値挙動は一切変わらない。
    def _edit_model(self, mj_model: mujoco.MjModel) -> None:
        """mjx.put_model の直前に MjModel をいじるフック。"""

    def _motor_targets(self, applied: jax.Array, info: Dict[str, Any]) -> jax.Array:
        """方策出力 -> 関節角目標。基準姿勢からの残差。

        対角2脚バランス版 (env/balance.py) は、この基準姿勢をトロットの
        中間姿勢に差し替える。残差の中心が「保ちたい姿勢」でないと、
        学習初期の action≈0 がいきなり遊脚を床に落としてしまう。
        """
        return self._default_pose + applied * self._config.action_scale

    def _ground_height(self, xy: jax.Array) -> jax.Array:
        """xy (..., 2) 直下の地面高さ [m]。戻り値の shape は xy.shape[:-1]。"""
        return jp.zeros(xy.shape[:-1])

    def _spawn(self, rng: jax.Array, qpos: jax.Array) -> jax.Array:
        """リセット時の水平位置と地面高さ。平地版は原点のまま。"""
        del rng
        return qpos

    # ---------- センサ ----------
    def _sensor(self, data, name):
        return mjx_env.get_sensor_data(self.mj_model, data, name)

    def _gravity(self, data):
        """胴体座標系での重力方向単位ベクトル。実機では IMU から推定する量。"""
        return -self._sensor(data, consts.UPVECTOR_SENSOR)

    # ---------- reset ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, k = jax.random.split(rng)
        qpos = qpos.at[0:3].set(
            qpos[0:3] + jax.random.uniform(k, (3,), minval=-0.01, maxval=0.01)
        )
        # fold_in で派生キーを作る。split して rng を消費すると平地版の乱数列が
        # 従来と変わってしまい、過去の run と比較できなくなる。
        qpos = self._spawn(jax.random.fold_in(k, 0), qpos)
        rng, k = jax.random.split(rng)
        yaw = jax.random.uniform(k, (), minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(
            pg_math.quat_mul(
                qpos[3:7], pg_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
            )
        )
        rng, k = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(k, (12,), minval=0.9, maxval=1.1)
        )
        rng, k = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(k, (6,), minval=-0.2, maxval=0.2))

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
        rng, kb1, kb2, kb3 = jax.random.split(rng, 4)
        ic = self._config.imu_config
        pc = self._config.pert_config
        info: Dict[str, Any] = {
            "rng": rng,
            "command": jax.random.uniform(k1, (3,), minval=-self._cmd_a,
                                          maxval=self._cmd_a),
            "steps_until_next_cmd": jp.round(
                jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            # 観測履歴のリングバッファ。buf[0] が 1 ステップ前のフレーム。
            # **reset では詰めない** (全ゼロのまま返す)。実機も電源投入直後は
            # バッファが空なので、その立ち上がりを学習時にも通す。これで
            # sim と実機の観測が最初の 1 ステップから完全に一致する
            # (scripts/check_deploy.py が数値で突き合わせる)。
            "obs_hist": jp.zeros((self._hist_frames, HIST_FRAME_DIM)),
            # 遅延バッファ: 実機の通信・サーボ応答遅れを模擬
            "action_buf": jp.zeros((self._max_delay + 1, self.mjx_model.nu)),
            "delay": jax.random.randint(
                k3, (), self._config.action_delay.min_steps,
                self._config.action_delay.max_steps + 1),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            # 胴体直下の地面高さ。climb 報酬が前ステップとの差を取る
            "last_ground": jp.mean(
                self._ground_height(data.site_xpos[self._feet_site_id][..., :2])),
            "ground_delta": jp.zeros(()),
            # 歩容位相 (内部クロック。実機でも同じ式で再現できる)
            "phase": jp.zeros(()),
            "gait_freq": jax.random.uniform(k4, (), minval=self._config.gait_freq[0],
                                            maxval=self._config.gait_freq[1]),
            "steps_until_next_pert": jp.round(
                jax.random.uniform(k5, minval=pc.kick_wait_times[0],
                                   maxval=pc.kick_wait_times[1]) / self.dt
            ).astype(jp.int32),
            "pert_duration": jp.round(
                jax.random.uniform(k6, minval=pc.kick_durations[0],
                                   maxval=pc.kick_durations[1]) / self.dt
            ).astype(jp.int32),
            "pert_steps": jp.zeros((), jp.int32),
            "pert_dir": jp.zeros(3),
            "pert_mag": jp.zeros(()),
            # IMU の定常バイアス (電源投入ごとに変わる個体差・温度ドリフト)
            "gyro_bias": jax.random.uniform(
                kb1, (3,), minval=-1, maxval=1) * ic.level * ic.gyro_bias,
            "accel_bias": jax.random.uniform(
                kb2, (3,), minval=-1, maxval=1) * ic.level * ic.accel_bias,
            "tilt_bias": jax.random.uniform(
                kb3, (3,), minval=-1, maxval=1) * ic.level * ic.gravity_tilt_bias,
        }
        metrics = {f"reward/{k}": jp.zeros(())
                   for k in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())
        metrics["linkage_violation"] = jp.zeros(())
        for k in ("track/speed", "track/cmd_speed", "track/lin_err", "track/ang_err",
                  "hw/tau_sat", "hw/tau_max", "hw/flight", "hw/duty"):
            metrics[k] = jp.zeros(())

        obs = self._obs_reset(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ---------- step ----------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        if self._config.pert_config.enable:
            state = self._maybe_perturb(state)
            info = state.info

        # 遅延適用: buf[0] が最新。delay ステップ前の指令を実際に出す
        buf = jp.roll(info["action_buf"], 1, axis=0).at[0].set(action)
        info["action_buf"] = buf
        applied = jp.where(
            self._config.action_delay.enable, buf[info["delay"]], action
        )
        motor_targets = self._motor_targets(applied, info)
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        contact = data.sensordata[self._feet_contact_adr] > 0
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        feet_xpos = data.site_xpos[self._feet_site_id]
        # 足の滞空高さは「その足の真下の地面」から測る。段差の上では
        # ワールド z で測ると段の高さぶん常に浮いている判定になる。
        p_fz = feet_xpos[..., -1] - self._ground_height(feet_xpos[..., :2])
        info["swing_peak"] = jp.maximum(info["swing_peak"], p_fz)

        # 足元の地面高さの変化。climb 報酬の材料。
        # 対称にクリップするのが肝心で、こうすると1エピソードの総和が
        # 「最終高さ - 初期高さ」に telescoping する。片側だけ (上りのみ加点) に
        # すると段の縁で上下動を繰り返すだけで無限に稼げてしまう。
        # クリップ幅は 1 ステップ (20ms) で起きうる最大の地面高さ変化。
        ground = jp.mean(self._ground_height(feet_xpos[..., :2]))
        info["ground_delta"] = jp.clip(ground - info["last_ground"], -0.01, 0.01)
        info["last_ground"] = ground
        info["phase"] = jp.fmod(info["phase"] + info["gait_freq"] * self.dt, 1.0)

        done = self._get_termination(data, info)
        # 報酬は行動更新の前に計算する (action_rate が a_t - a_{t-1} になるように)
        rewards = self._get_reward(data, action, info, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k]
                   for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, -10.0, 10000.0)

        # 観測は行動更新の「後」に作る。
        # Go1 の実装は obs を更新前に作るので last_act が a_{t-1} になるが、
        # PrintPupper には関節エンコーダが無く、行動履歴が唯一の自己受容感覚なので、
        # 「いま出したばかりの指令 a_t を方策が知らない」のは致命的な欠損になる。
        # ここでは a_t と a_{t-1} が観測に入るようにする。
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        obs = self._get_obs(data, info)
        # 履歴は観測を作った **後** に詰める。こうすると
        #   obs(t) が見る履歴 = フレーム t-1, t-2, ...   (t 自身は含まない)
        # になり、実機側 (env/deploy.py: step() -> commit()) と同じ順序になる。
        self._push_hist(info, obs["state"])

        info["steps_until_next_cmd"] -= 1
        info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
        info["command"] = jp.where(
            info["steps_until_next_cmd"] <= 0,
            self.sample_command(k1, info["command"]), info["command"])
        info["steps_until_next_cmd"] = jp.where(
            done | (info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(k2) * 5.0 / self.dt).astype(jp.int32),
            info["steps_until_next_cmd"])
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(info["swing_peak"])
        state.metrics["linkage_violation"] = infeasibility(data.qpos[7:])
        # 報酬に埋もれない生の追従指標。「立ち止まって報酬を稼ぐ」失敗を検知する。
        lv = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        state.metrics["track/speed"] = jp.linalg.norm(lv[:2])
        state.metrics["track/cmd_speed"] = jp.linalg.norm(info["command"][:2])
        state.metrics["track/lin_err"] = jp.linalg.norm(info["command"][:2] - lv[:2])
        state.metrics["track/ang_err"] = jp.abs(
            info["command"][2] - self._sensor(data, consts.GYRO_SENSOR)[2])
        # 実機の寿命を決める指標。報酬とは別に生で追う。
        state.metrics["hw/tau_sat"] = jp.mean(
            (jp.abs(data.actuator_force) > self._stall * 0.995).astype(jp.float32))
        state.metrics["hw/tau_max"] = jp.max(jp.abs(data.actuator_force))
        state.metrics["hw/flight"] = (jp.sum(contact) == 0).astype(jp.float32)
        state.metrics["hw/duty"] = jp.mean(contact.astype(jp.float32))

        done = done.astype(reward.dtype)
        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    def _get_termination(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        fall = self._gravity(data)[-1] > -0.3          # 45度以上傾いたら終了
        trunk = data.sensordata[self._trunk_contact_adr] > 0
        return fall | trunk

    # ---------- 観測 ----------
    def _imu(self, data, info):
        """実機の 6軸 IMU が返すものを模擬する。

        返り値:
          gyro_m    ジャイロ生値 (バイアス + 白色ノイズ)
          accel_m   加速度計生値 = 比力 (バイアス + 白色ノイズ)
          grav_m    オンボード姿勢推定が出す重力方向 (単位ベクトル)

        姿勢推定の誤差モデル: 相補フィルタは短期はジャイロを信じるが、
        機体加速度を完全には除去できない。加速度計のみのチルト推定
        (-f/|f|) と真値の内分で近似し、係数 gravity_accel_coupling で
        「どれだけ加速度に騙されるか」を表す。歩行中は常に加速度が乗るので、
        この項を入れないと実機で効かない姿勢情報に方策が依存してしまう。
        """
        ic = self._config.imu_config
        info["rng"], k1, k2, k3 = jax.random.split(info["rng"], 4)

        gyro = self._sensor(data, consts.GYRO_SENSOR)
        accel = self._sensor(data, consts.ACCELEROMETER_SENSOR)   # 比力
        grav_true = self._gravity(data)                            # -upvector

        gyro_m = (gyro + info["gyro_bias"]
                  + jax.random.normal(k1, (3,)) * ic.level * ic.gyro_noise)
        accel_m = (accel + info["accel_bias"]
                   + jax.random.normal(k2, (3,)) * ic.level * ic.accel_noise)

        # 加速度計のみで出したチルト推定
        grav_from_accel = -accel_m / (jp.linalg.norm(accel_m) + 1e-6)
        a = ic.gravity_accel_coupling
        grav = (1.0 - a) * grav_true + a * grav_from_accel
        grav = grav + info["tilt_bias"] + (
            jax.random.normal(k3, (3,)) * ic.level * ic.gravity_noise)
        grav_m = grav / (jp.linalg.norm(grav) + 1e-6)
        return gyro_m, accel_m, grav_m

    def _push_hist(self, info: Dict[str, Any], obs_state: jax.Array) -> None:
        """観測履歴のリングバッファに 1 フレーム詰める。

        フレームは **いま作った観測から切り出す** (IMU を引き直さない)。
        _imu() は毎回ノイズを引くので、作り直すと観測に載った値と履歴に
        残る値がずれてしまう。obs_state の先頭 9 次元はちょうど
        [gyro, accel*scale, gravity] なので、そこに今出した行動を足すだけでよい。
        """
        if self._hist_len <= 0:
            return
        frame = jp.concatenate([obs_state[:9], info["last_act"]])
        info["obs_hist"] = jp.roll(info["obs_hist"], 1, axis=0).at[0].set(frame)

    def _obs_reset(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """reset 用の観測。履歴バッファを「いまの姿勢のフレーム」で埋めてから作る。

        **ゼロ埋めのまま始めてはいけない。** 観測は running_statistics で
        正規化されてネットワークに入るが、重力 z は mean -0.9935 / std 0.0188
        なので、ゼロの履歴フレームは **正規化後に +52.7σ** という桁外れの入力に
        なる (e40 の normalizer で実測)。

        拡張した入力重みは初期値ゼロなので方策の出力自体は変わらない。
        壊れるのは勾配のほうで、**重みに乗る勾配は入力に比例する**ため、
        最初の更新で巨大な値が書き込まれて収束済みの方策が飛ぶ。
        実測 (8M step, --env terrain, e40 から継続):
            履歴なし       報酬 8.08 -> 9.70   (安定)
            履歴ゼロ埋め   報酬 6.96 -> -33.2  (崩壊)
        エピソード頭の 0.8 秒ぶんだけの話なのに、全体が壊れる。

        実機も電源投入時に 1 回目の IMU 読み値でバッファを埋める
        (env/deploy.py の ObsBuilder)。「静止したまま 0.8 秒ぶんの履歴が
        溜まっていた」という、分布の内側の状態から始めることになる。
        """
        obs = self._get_obs(data, info)
        if self._hist_len > 0:
            frame = jp.concatenate([obs["state"][:9], info["last_act"]])
            info["obs_hist"] = jp.tile(frame, (self._hist_frames, 1))
            obs = self._get_obs(data, info)     # 埋めたバッファで作り直す
        return obs

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        gyro_m, accel_m, grav_m = self._imu(data, info)
        ic = self._config.imu_config
        ph = 2 * jp.pi * info["phase"]

        # actor: 実機に載っているのは IMU だけ。関節エンコーダも接地センサも無い。
        # 関節の状態は「自分が出した指令の履歴」からしか知れない。
        state = jp.hstack([
            gyro_m,                        # 3  IMU ジャイロ [rad/s]
            accel_m * ic.accel_obs_scale,  # 3  IMU 加速度計 (比力) [m/s^2]
            grav_m,                        # 3  オンボード姿勢推定の重力方向
            info["command"],               # 3  速度コマンド (ユーザ入力)
            info["last_act"],              # 12 前ステップの自分の出力
            info["last_last_act"],         # 12 2ステップ前 (指令の変化率)
            jp.array([jp.cos(ph), jp.sin(ph)]),   # 2 歩容位相 (内部クロック)
        ])
        # 観測履歴スタック。現在のフレーム (上の 38 次元) の **後ろ** に足す。
        # 先頭 38 次元の意味を変えないので、38 次元で学習した重みは
        # 「履歴部の入力重みを 0 で埋める」だけで完全に等価なまま移せる
        # (scripts/adapt_ckpt.py)。
        if self._hist_len > 0:
            state = jp.concatenate(
                [state, info["obs_hist"][self._hist_idx].ravel()])

        # critic: 実機には無い量も使ってよい (非対称 actor-critic)
        joint_angles = data.qpos[7:]
        privileged = jp.hstack([
            state,
            self._sensor(data, consts.GYRO_SENSOR),           # ノイズ無しの真値
            self._sensor(data, consts.ACCELEROMETER_SENSOR),
            self._gravity(data),
            self._sensor(data, consts.LOCAL_LINVEL_SENSOR),
            self._sensor(data, consts.GLOBAL_ANGVEL_SENSOR),
            joint_angles - self._default_pose,                # 実機には無い関節角
            data.qvel[6:],                                    # 実機には無い関節速度
            data.actuator_force,
            (data.sensordata[self._feet_contact_adr] > 0).astype(jp.float32),
            info["feet_air_time"],
            info["gyro_bias"], info["accel_bias"], info["tilt_bias"],
            jp.array([info["gait_freq"]]),
            jp.array([infeasibility(joint_angles)]),
        ])
        return {"state": state, "privileged_state": privileged}

    # ---------- 報酬 ----------
    def _get_reward(self, data, action, info, done, first_contact, contact):
        rc = self._config.reward_config
        cmd = info["command"]
        local_v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        gyro = self._sensor(data, consts.GYRO_SENSOR)
        gravity = self._gravity(data)
        joint_angles = data.qpos[7:]

        lin_err = jp.sum(jp.square(cmd[:2] - local_v[:2]))
        ang_err = jp.square(cmd[2] - gyro[2])
        cmd_norm = jp.linalg.norm(cmd)
        moving = cmd_norm > 0.02

        feet_vel = data.sensordata[
            (self._feet_linvel_adr[:, None] + np.arange(3)[None])
        ]
        slip = jp.sum(jp.square(feet_vel[:, :2]) * contact[:, None])

        # 位相クロックが指示する足先高さに追従させる。
        # get_rz は位相角 [-pi, pi] に対する遊脚高さプロファイル (中央で最高)。
        # 足先 site は接地時でも足球半径ぶん浮いているので、半径を引いて比較する。
        # 高さは「その足の真下の地面」から測る。ワールド z のままだと、段の上に
        # 乗った足に地形高さぶんの誤差が定常的に乗り、方策には消しようがない。
        # 平地では _ground_height が厳密に 0 を返すので挙動は変わらない。
        phi = 2 * jp.pi * info["phase"] + self._foot_phase_offset
        phi = (phi + jp.pi) % (2 * jp.pi) - jp.pi
        rz = pg_gait.get_rz(phi, swing_height=rc.max_foot_height)
        feet_xpos = data.site_xpos[self._feet_site_id]
        foot_z = (feet_xpos[:, 2] - consts.FOOT_RADIUS
                  - self._ground_height(feet_xpos[:, :2]))
        feet_phase = jp.exp(-jp.sum(jp.square(foot_z - rz)) / rc.feet_phase_sigma)
        feet_phase *= moving              # 停止指令時は足踏みを要求しない

        # 遊脚期の窓。rz > 0 をマスクに使ってはいけない: get_rz は位相の 98% で
        # 正になり (立脚期に 0 で留まる区間が無い)、4脚同時に浮いた姿勢 =
        # 飛翔期を要求してしまう。gait.yaml の stance_duration から明示的に取る。
        # get_rz は phi=0 で最高なので、遊脚期は phi=0 を中心とした窓になる。
        in_swing = jp.abs(phi) < jp.pi * (1.0 - consts.STANCE_DURATION)

        # トルク軟限界とフライト期
        over = jp.clip(jp.abs(data.actuator_force)
                       - self._stall * rc.torque_soft_limit_frac, 0.0, None)
        flight = (jp.sum(contact) == 0).astype(jp.float32)

        out_of_limits = jp.sum(
            jp.clip(self._soft_lowers - joint_angles, 0.0, None)
            + jp.clip(joint_angles - self._soft_uppers, 0.0, None)
        )
        # 胴体高さは「4足の真下の地面の平均」から測る。ワールド z 絶対値で
        # 測ると、段に乗った瞬間に base_height ペナルティが入って登らなくなる。
        ground = jp.mean(self._ground_height(data.site_xpos[self._feet_site_id][..., :2]))
        base_h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS - ground

        return {
            "tracking_lin_vel": jp.exp(-lin_err / rc.tracking_sigma),
            "tracking_ang_vel": jp.exp(-ang_err / rc.tracking_sigma_ang),
            "lin_vel_z": jp.square(local_v[2]),
            "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
            "orientation": jp.sum(jp.square(gravity[:2])),
            "base_height": jp.square(base_h - rc.base_height_target),
            "feet_phase": feet_phase,
            # 足上げ不足だけを罰する。feet_phase の目標 rz は立脚期に 0 で留まる
            # 区間が無く、トロット位相を入れると 4本同時に浮いた姿勢を要求する
            # (= 飛翔期)。flight ペナルティと矛盾するので、feet_phase だけでは
            # 「高く上げろ」という圧力を出せない。段差越えに必要な足上げは
            # この項が駆動する。
            #
            # 以前は first_contact (接地の瞬間) でしか評価していなかった。これは
            # 全脚合計で毎秒2〜3回しか発火せず、毎ステップ効く tracking に対して
            # 弱すぎる。実測: 重みを -60 にし目標を 85->105mm に上げても遊脚高さは
            # 21mm から 1mm も動かなかった (s50)。さらに「足を動かさなければ接地
            # イベント自体が起きない」ため、罰を回避する縮退解の入口にもなっていた。
            # バランス版 (env/balance.py の swing_clearance) と同じく毎ステップ
            # 評価に変える。swing_peak は「今回の遊脚での最高到達点」なので、
            # 目標に届いた時点で罰が 0 になり、矩形波のような軌道は要求しない。
            # 接地したまま擦っている脚は swing_peak が毎ステップ 0 に戻るので、
            # 遊脚窓の間ずっと満額の罰を受ける。
            # info["swing_peak"] は step() で地面相対に補正済み。
            # swing_peak は足球中心の高さなので、feet_phase と揃えて半径を引く。
            "feet_clearance": jp.sum(
                jp.clip(rc.max_foot_height
                        - (info["swing_peak"] - consts.FOOT_RADIUS), 0.0, None)
                * in_swing) * moving,
            "feet_slip": slip,
            # 段を登れたこと自体への加点。これまでの報酬は速度追従・足上げ・
            # 正則化だけで、「登坂」は "登れないと前に進めないので tracking が
            # 下がる" という間接信号しか無かった。段の前で止まっても失う報酬は
            # わずかで、段を避けて平地を歩くほうが安全で高得点になっていた。
            # 実測: 40mm/50mm 地形で 3 通り試して全て学習が停止した (d40/h50/e40)。
            "climb": info["ground_delta"] / self.dt * moving,
            "flight": flight,
            "torques": jp.sum(jp.square(data.actuator_force)),
            "torque_limit": jp.sum(jp.square(over)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            # 股関節(外転)だけを基準姿勢に引き戻す。全関節に掛けると「脚を振る」
            # こと自体を罰してしまい、静止が最適解になる。
            "pose": jp.sum(jp.square(
                (joint_angles - self._default_pose)[self._hip_idx])),
            "dof_pos_limits": out_of_limits,
            "stand_still": jp.sum(jp.abs(joint_angles - self._default_pose))
            * (cmd_norm < 0.02),
            "termination": done.astype(jp.float32),
            "linkage_infeasible": infeasibility(joint_angles),
        }

    # ---------- コマンド / 外乱 ----------
    def sample_command(self, rng: jax.Array, prev: jax.Array) -> jax.Array:
        k1, k2 = jax.random.split(rng)
        c = jax.random.uniform(k1, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        keep = jax.random.bernoulli(k2, self._cmd_b, (3,))
        return jp.where(keep, c, jp.zeros(3))

    def _maybe_perturb(self, state: mjx_env.State) -> mjx_env.State:
        info = state.info
        info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
        start = info["steps_until_next_pert"] <= 0
        pc = self._config.pert_config
        new_dir = jax.random.normal(k1, (3,)).at[2].set(0.0)
        new_dir = new_dir / (jp.linalg.norm(new_dir) + 1e-6)
        new_mag = jax.random.uniform(k2, minval=pc.velocity_kick[0],
                                     maxval=pc.velocity_kick[1])
        info["pert_dir"] = jp.where(start, new_dir, info["pert_dir"])
        info["pert_mag"] = jp.where(start, new_mag, info["pert_mag"])
        info["pert_steps"] = jp.where(start, 0, info["pert_steps"])
        active = info["pert_steps"] < info["pert_duration"]
        apply = active & ~start | start
        qvel = state.data.qvel
        qvel = qvel.at[:3].add(
            jp.where(apply & (info["steps_until_next_pert"] <= 0),
                     info["pert_dir"] * info["pert_mag"], 0.0))
        info["pert_steps"] = jp.where(active, info["pert_steps"] + 1,
                                      info["pert_steps"])
        info["steps_until_next_pert"] = jp.where(
            active & ~start, info["steps_until_next_pert"],
            info["steps_until_next_pert"] - 1)
        info["steps_until_next_pert"] = jp.where(
            info["pert_steps"] >= info["pert_duration"],
            jp.round(jax.random.uniform(
                k1, minval=pc.kick_wait_times[0],
                maxval=pc.kick_wait_times[1]) / self.dt).astype(jp.int32),
            info["steps_until_next_pert"])
        return state.replace(data=state.data.replace(qvel=qvel))

    # ---------- properties ----------
    @property
    def xml_path(self) -> str: return self._xml_path
    @property
    def action_size(self) -> int: return self._mjx_model.nu
    @property
    def mj_model(self) -> mujoco.MjModel: return self._mj_model
    @property
    def mjx_model(self) -> mjx.Model: return self._mjx_model
