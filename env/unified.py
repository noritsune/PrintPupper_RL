"""PrintPupper 段差歩行 + 転倒復帰の両立タスク (1つの方策で両方やる)。

やること: 段差 40mm を越えて速度コマンドに追従して歩き、**転んだらその場で
自力で起き上がって歩行に戻る**。これまで e40 (段差歩行) と rec9 (転倒復帰) の
2つの方策を scripts/recover_handoff.py で外から切り替えていたものを、
1つのネットワークに統合する。

## 最初に当たる壁: 可動域 (action_scale) が両立しない

歩行の関節指令は「基準姿勢 + action * 0.25 rad」。方策出力は tanh で [-1,1] に
収まるので、**可動域は厳密に「基準姿勢 ± action_scale」**になる。

  - 歩行に 1.6 を使う: 出力 1 目盛あたり 6.4 倍粗くなる。段差越えの足上げは
    0.2 rad 程度の精度で効くので、歩容がそのぶん雑になる。
  - 復帰に 0.25 を使う: **原理的に不可能**。仰向けの機体は胴体箱の上面
    (機体フレーム z=+57mm) で床に乗り、足先を床に着けるには 73度のロールが要る
    (env/recover.py の実測表)。0.25 では脚を横に開けず、そのロールを作れない。

つまり「どちらか一方の定数」では解けない。ここでの解は

  **可動域そのものを胴体の傾きの関数にする。**

      up = upvector z  (+1 正立 / 0 横倒し / -1 仰向け)
      a  = smoothstep(up; 0.50, 0.90)          # 1 正立 / 0 転倒
      action_scale(up) = 1.6 + (0.25 - 1.6) * a

up は実機の IMU が出す重力方向そのもので、**外受容センサを増やさずに実機で
そのまま同じ式が書ける** (env/deploy.py の ObsBuilder / step_targets)。
帯 (0.50-0.90 = 傾き 60度-26度) は

  - 段差 40mm を越えるときの胴体の傾きは実測で最大 20度 (up=0.94) なので
    **歩行中は帯に入らない** = 歩行時の可動域は 0.25 ちょうど。
  - 仰向け・横倒しは up <= 0 なので **復帰時は 1.6 ちょうど**。

という幅に取ってある。IMU の姿勢推定誤差 (tilt_bias 0.03rad + ノイズ) は
数度なので帯をまたがない。

この設計の効き目は初期値にも出る。上端が e40 の、下端が rec9 の写像と
**数値的に完全に一致する**ので、どちらの重みを --restore しても、その方策が
得意な領域では意味がそのまま保たれる (片方の技能をゼロから学び直さない)。

同じ理由で観測も両者に揃える: 転倒中 (a < 0.5) は速度コマンドを 0、歩容位相を
0 に固定する。rec9 は「コマンド 0 / 位相 0 固定」で学習しているので、転倒中の
観測分布が rec9 の学習時と一致する。実機側も「転んでいる間はスティック入力を
無視して位相を止める」という自明な規則で再現できる。

## そのほかの合流点

- **モデル**: model/pupper_unified.xml = 地形 hfield (terrain 版) + 大腿・下腿の
  衝突カプセル (recover 版)。歩行版の XML は脚が床をすり抜けるので、脚で床を
  押す復帰動作が実機で成立しないものになる。カプセルは 4脚接地の姿勢では
  床に触れない (足球より 12.6mm 高い) ので、接地センサの意味は歩行版と同じ。
- **途中終了しない**: 歩行版は「45度傾いたら終了」だが、この課題では転倒は
  入口であって失敗ではない。打ち切るのは地形の外に出たときだけ。
  転倒を避ける圧力は、罰ではなく **転ぶと歩行報酬 (3-4/step) を失うこと**と
  upright 項で与える。
- **初期姿勢**: 半分は地形上の通常スポーン (歩行)、半分は仰向け / 横倒し /
  伏せ (復帰)。復帰も段差地形の上で起こすので、平地限定にならない。
- **報酬**: 歩行系は a で、復帰系は (1-a) でゲートする。両方が同時に効くのは
  帯の中 (傾き 26-60度) だけ。正則化は両者で重みが違う (復帰は大振りするので
  弱くする) ので、a で重みを補間する。

詳しい経緯は OUTPUTS.md の「段差歩行と転倒復帰を1つの方策に統合する」。
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as pg_math
from mujoco_playground._src import mjx_env

from . import constants as consts
from .joystick import PupperJoystick
from .terrain import PupperTerrain, terrain_config


# 歩行版の報酬のうち「正立しているときだけ意味がある」項。a でゲートする。
WALK_GATED = (
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
    "orientation", "base_height", "feet_phase", "feet_clearance", "climb",
    "feet_slip", "flight", "pose", "stand_still",
)
# 正則化は両モードで必要だが重みが違う。値は「転倒側の重み / 歩行側の重み」で、
# 歩行版 (env/joystick.py) と復帰版 (env/recover.py) の既定値の比そのもの:
#   torques      -0.002 -> -0.0005   action_rate -0.02 -> -0.01
#   energy       -0.001 -> -0.0005   torque_limit -1.0 -> -0.01
# 復帰は全可動域を使う大振りなので、歩行と同じ重みだと「仰向けのまま動かない」
# が最適になる (env/recover.py の注記)。逆に歩行側を緩めると歩容が荒れる。
REG_RATIO = dict(torques=0.25, action_rate=0.5, energy=0.5)


def unified_config(step_height: float = 0.040) -> config_dict.ConfigDict:
    """段差歩行 (e40) と転倒復帰 (rec9) の設定を合流させたもの。

    歩行側の既定値は e40 の run と同じにしてある (段差 40mm / 足上げ目標
    据え置き 85mm / 歩容周期 0.7-1.2Hz / スポーン半径 4.0m)。
    """
    cfg = terrain_config(step_height=step_height)

    # --- e40 の run が使った値をここに畳み込む (OUTPUTS.md の「効いた一手 その3」) ---
    # 足上げ目標を地形の段差に連動させると 40mm 地形では 95mm に跳ね上がり、
    # 方策は足上げに過剰投資して胴体の扱いを失う。据え置きが正解だった。
    cfg.reward_config.max_foot_height = 0.085
    cfg.gait_freq = [0.7, 1.2]
    cfg.terrain.spawn_radius = 4.0
    cfg.terrain.difficulty = 1.0

    # 復帰 (~1.3s) の後に歩行を十分評価できる長さ。歩行版と同じ 20s。
    cfg.episode_length = 1000
    # cfg.action_scale は使わない (傾きの関数にする。下の _act_scale)。
    # 誤って参照されたときに歩行側の値になるよう、歩行側を入れておく。
    cfg.action_scale = consts.WALK_ACTION_SCALE
    # 接触: 足球4 + 脚カプセル8 + 胴体箱。仰向けでは胴体箱が最大4点を出す。
    cfg.naconmax = 16 * 4096

    cfg.unified = config_dict.create(
        act_scale_walk=consts.WALK_ACTION_SCALE,
        act_scale_fallen=consts.RECOVER_ACTION_SCALE,
        gate_lo=consts.GATE_UP_LO,          # up がこれ以下なら完全に「転倒」
        gate_hi=consts.GATE_UP_HI,          # up がこれ以上なら完全に「正立」
        # 観測 (コマンド / 歩容位相) を転倒側に切り替える閾値。a のしきい値。
        obs_switch=0.5,
        # 転倒中のトルク軟限界。歩行は 0.5*stall だが、復帰は瞬間的に必ず
        # 飽和するので 0.8*stall まで緩める (env/recover.py の実測)。
        torque_soft_limit_frac_fallen=0.8,
        # 転倒中の torque_limit 罰の倍率 (歩行 -1.0 に対し復帰 -0.01)。
        torque_limit_ratio=0.01,
    )

    # --- 初期姿勢 ---
    # p_walk で歩行スポーン、残りを転倒姿勢に割る。転倒側の内訳 (p_flip /
    # p_side / p_prone) は **転倒側の中での条件付き確率**。
    cfg.recover = config_dict.create(
        p_walk=0.50,
        p_flip=0.60,                    # 仰向け (roll ≈ 180度)
        p_side=0.20,                    # 横倒し (roll ≈ ±90度)
        p_prone=0.20,                   # 伏せ (正立だが脚を前に投げ出した姿勢)
        prone_delta=[0.0, -0.7, -0.6],
        roll_jitter=0.6,
        pitch_jitter=0.4,
        pose_jitter=0.35,
        clearance=0.002,
        init_vel=0.10,
        init_avel=0.50,
        # --- 「復帰した」の判定 (env/recover.py と同一) ---
        upright_cos=0.85,
        height_tol=0.030,
        feet_needed=3,
        settle_lin=0.10,
        settle_ang=1.00,
        hold_steps=25,
        height_sigma=0.004,
        trunk_gate=True,
        p_stuck=0.0,
    )

    s = cfg.reward_config.scales
    # --- 復帰系 (env/recover.py の rec9 の重み) ---
    # upright だけはゲートしない。正立している間は +4/step の定数になり、
    # 「転ぶと失うもの」= 転倒を避ける圧力そのものになる (歩行版の
    # termination 罰の代わり。この課題では途中終了しないので罰では書けない)。
    s.upright = 4.0
    # **転んでいること自体への罰**。歩行版は「45度傾いたら終了」で、転倒は
    # 残り時間ぶんの報酬 (9/step x 十数秒) をまるごと失う暗黙の巨大コスト
    # だった。両立版は転倒で終了しないので、そのコストが消える。すると
    # **転倒が「歩行の罰からの逃げ場」になる**: run u4 は 100M step で報酬を
    # 21.7 -> 35.4 に伸ばしながら、その中身は
    #   feet_clearance +1147 / torque_limit +981 (= 足を上げずトルクも出さない)
    #   upright -743 / trunk_contact -513 (= 起きるのをやめた)
    # で、正立時間は 83% -> 50% に落ちた。報酬は上がって挙動は悪化する典型。
    # 歩行側の重みは e40 で調整済みなので触らず、**正立時に 0 で転倒時にだけ
    # 立ち上がる罰**を 1 項足して終了罰の代わりにする。ゲート a と同じ形なので
    # 復帰の勾配 (upright) の形は変えない。
    s.fallen = -6.0
    s.height = 3.0
    s.height_low = -25.0
    s.feet_down = 4.0
    s.recovered = 4.0
    s.rock = 0.0
    s.stand_pose = -3.0
    s.settle = -0.3
    s.trunk_contact = -2.0
    # rec9 は rec8 から linkage_infeasible を -5 -> -20 に上げて実現不能率を
    # 3.34% -> 1.11% に下げた run。復帰動作は関節を端まで使うので歩行側の
    # -5 では足りない。歩行中の実現不能率はもともと 0 に近いので副作用は無い。
    s.linkage_infeasible = -20.0
    return cfg


class PupperUnified(PupperTerrain):
    """段差歩行 + 転倒復帰。観測次元は歩行版と同一 (actor 38)。"""

    # 地形 hfield と脚の衝突形状を両方持つ XML。
    XML = consts.UNIFIED_XML

    def __init__(
        self,
        config: config_dict.ConfigDict = unified_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def _post_init(self) -> None:
        super()._post_init()
        m = self._mj_model
        self._trunk_geom_id = m.geom("trunk_col").id
        self._trunk_size = jp.array(m.geom_size[self._trunk_geom_id])
        self._box_corners = jp.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=jp.float32)
        rc = self._config.recover
        self._prone_delta = jp.array(
            np.tile(np.asarray(rc.prone_delta, np.float32), 4))
        self._hold_needed = int(rc.hold_steps)
        self._feet_needed = int(rc.feet_needed)
        self._stuck = None          # load_stuck_states() で入る

    # ---------- 詰まり状態からの初期化 (env/recover.py と同じ) ----------
    def load_stuck_states(self, path) -> int:
        """前の run が実際に詰まった状態 (.npz) を初期姿勢の候補に加える。

        行動履歴も一緒に復元するのが肝心 (実機に関節エンコーダが無いので、
        観測の自己受容感覚は行動履歴しかない)。xy は地形上のスポーン位置に
        置き直し、高さは reset 側の「床に置き直す」処理に任せる。
        """
        d = np.load(path)
        self._stuck = (jp.array(d["qpos"], jp.float32),
                       jp.array(d["qvel"], jp.float32),
                       jp.array(d["act"], jp.float32))
        return int(d["qpos"].shape[0])

    # ---------- 可動域のゲート ----------
    def _gate(self, up: jax.Array) -> jax.Array:
        """up (upvector z) -> 1 正立 / 0 転倒 の滑らかなゲート。"""
        uc = self._config.unified
        t = jp.clip((up - uc.gate_lo) / (uc.gate_hi - uc.gate_lo), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)      # smoothstep (端で微分 0)

    def _act_scale(self, up: jax.Array) -> jax.Array:
        uc = self._config.unified
        a = self._gate(up)
        return uc.act_scale_fallen + (uc.act_scale_walk - uc.act_scale_fallen) * a

    def _motor_targets(self, applied: jax.Array, info: Dict[str, Any]) -> jax.Array:
        """方策出力 -> 関節角目標。可動域は info["act_scale"] (傾きの関数)。

        info["act_scale"] は 1 ステップ前の姿勢から作る。実機でも「直前に読んだ
        IMU の値で今の指令を作る」ので、この 1 ステップの遅れは実機と同じ。
        """
        return self._default_pose + applied * info["act_scale"]

    # ---------- 観測 ----------
    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """転倒中はコマンドと歩容位相を 0 に固定してから歩行版の観測を作る。

        rec9 は「コマンド 0 / 位相 0 固定」で学習しているので、転倒中の観測が
        rec9 の学習時の分布と一致する (--restore した重みがそのまま効く)。
        実機側も「転んでいる間はスティック入力を無視し、位相クロックを止める」
        という規則で同じことができる。
        """
        a = self._gate(-self._gravity(data)[2])
        up_now = a > self._config.unified.obs_switch
        info = dict(info)
        info["command"] = jp.where(up_now, info["command"], jp.zeros(3))
        info["phase"] = jp.where(up_now, info["phase"], jp.zeros(()))
        obs = super()._get_obs(data, info)
        # 壊れた env の観測が方策に入ると、その 1 本のためにネットワーク全体が
        # NaN になる (_get_termination の注記)。done は同じステップで立つので、
        # ここは「最後の 1 ステップを無害な値にする」ためのもの。
        return {k: jp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                for k, v in obs.items()}

    # ---------- reset ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # 地形版の reset に歩行スポーン (地形上のランダム位置) を作らせてから、
        # 確率 1-p_walk で姿勢だけ転倒姿勢に差し替える。xy はそのまま使うので、
        # 復帰も段差地形の上で起きる。
        state = super().reset(rng)
        info = state.info
        rc = self._config.recover

        info["rng"], k_mode, k_roll, k_pitch, k_yaw, k_q, k_v = jax.random.split(
            info["rng"], 7)

        u = jax.random.uniform(k_mode)
        walk = u < rc.p_walk
        # 転倒側の中での位置 (0-1)。p_walk=1 でも 0 除算しない。
        uf = jp.clip((u - rc.p_walk) / jp.maximum(1.0 - rc.p_walk, 1e-6), 0.0, 1.0)

        side_sign = jp.where(jax.random.bernoulli(k_yaw), 1.0, -1.0)
        roll0 = jp.where(uf < rc.p_flip, jp.pi,
                         jp.where(uf < rc.p_flip + rc.p_side,
                                  side_sign * jp.pi / 2, 0.0))
        prone = uf >= rc.p_flip + rc.p_side
        roll = roll0 + jax.random.uniform(
            k_roll, (), minval=-rc.roll_jitter, maxval=rc.roll_jitter)
        pitch = jax.random.uniform(
            k_pitch, (), minval=-rc.pitch_jitter, maxval=rc.pitch_jitter)
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        quat_fall = pg_math.quat_mul(
            pg_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw),
            pg_math.quat_mul(
                pg_math.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), pitch),
                pg_math.axis_angle_to_quat(jp.array([1.0, 0.0, 0.0]), roll)))

        # 実機が転んだ直後はサーボが最後の指令 (基準姿勢) を保持している。
        q_fall = jp.clip(
            self._default_pose
            + jp.where(prone, self._prone_delta, 0.0)
            + jax.random.uniform(
                k_q, (12,), minval=-rc.pose_jitter, maxval=rc.pose_jitter),
            self._soft_lowers, self._soft_uppers)

        qpos, qvel = state.data.qpos, state.data.qvel
        # 一旦十分に浮かせて FK を取る (地形の最大高さ + 余裕)。
        z_hi = self._t_max + 0.4
        qpos = jp.where(walk, qpos,
                        qpos.at[2].set(z_hi).at[3:7].set(quat_fall).at[7:].set(q_fall))
        kv1, kv2 = jax.random.split(k_v)
        qvel_fall = jp.zeros(self.mjx_model.nv)
        qvel_fall = qvel_fall.at[0:3].set(jax.random.uniform(
            kv1, (3,), minval=-rc.init_vel, maxval=rc.init_vel))
        qvel_fall = qvel_fall.at[3:6].set(jax.random.uniform(
            kv2, (3,), minval=-rc.init_avel, maxval=rc.init_avel))
        qvel = jp.where(walk, qvel, qvel_fall)

        # --- 詰まり状態への差し替え (転倒側の p_stuck の確率) ---
        # 床に置き直す前に入れておき、高さは下の処理にまとめて任せる。
        if self._stuck is not None and float(rc.p_stuck) > 0.0:
            sq, sv, sa = self._stuck
            info["rng"], k_i, k_u = jax.random.split(info["rng"], 3)
            i = jax.random.randint(k_i, (), 0, sq.shape[0])
            use = (~walk) & (jax.random.uniform(k_u) < rc.p_stuck)
            # 記録は平地 (env/recover.py) で取ったものなので xy は捨て、
            # 地形上のスポーン位置に移す。
            sqi = sq[i].at[0:2].set(qpos[0:2]).at[2].set(z_hi)
            qpos = jp.where(use, sqi, qpos)
            qvel = jp.where(use, sv[i], qvel)
            a0 = jp.where(use, sa[i], jp.zeros(self.mjx_model.nu))
            info["last_act"] = a0
            info["last_last_act"] = a0
            info["action_buf"] = jp.tile(a0, (info["action_buf"].shape[0], 1))

        data = mjx.forward(self.mjx_model, state.data.replace(
            qpos=qpos, qvel=qvel, ctrl=qpos[7:]))

        # --- 地形の上に置き直す ---
        # 床に当たりうるのは足球4個と胴体箱 (脚カプセルは常にそれらより内側)。
        # **地面高さは点ごとに引く**: 平地版のように z の最小値だけ見ると、
        # 段の上に胴体があって足が段の下にある配置で機体を地形へめり込ませる。
        feet = data.site_xpos[self._feet_site_id]
        feet_clear = (feet[:, 2] - consts.FOOT_RADIUS
                      - self._ground_height_max(feet[:, :2]))
        gpos = data.geom_xpos[self._trunk_geom_id]
        gmat = data.geom_xmat[self._trunk_geom_id].reshape(3, 3)
        corners = gpos + (self._box_corners * self._trunk_size) @ gmat.T
        corner_clear = corners[:, 2] - self._ground_height_max(corners[:, :2])
        drop = jp.minimum(jp.min(feet_clear), jp.min(corner_clear)) - rc.clearance
        qpos = qpos.at[2].add(-drop)
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos))

        info["hold"] = jp.zeros((), jp.int32)     # 連続で復帰条件を満たした数
        info["best_up"] = -self._gravity(data)[2]
        info["ever"] = jp.zeros(())               # 一度でも復帰したか
        info["act_scale"] = self._act_scale(-self._gravity(data)[2])
        # climb 報酬の基準も置き直した位置で取り直す。
        info["last_ground"] = jp.mean(
            self._ground_height(data.site_xpos[self._feet_site_id][..., :2]))

        metrics = dict(state.metrics)
        for k in ("rec/upright", "rec/height", "rec/feet", "rec/ok",
                  "rec/success", "rec/tilt", "rec/trunk", "rec/act_scale",
                  "rec/upright_frac"):
            metrics[k] = jp.zeros(())

        obs = self._obs_reset(data, info)
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    # ---------- step ----------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        data, info = state.data, state.info

        up, h, n_feet, ok = self._status(data, info)
        info["best_up"] = jp.maximum(info["best_up"], up)
        info["hold"] = jp.where(ok > 0, info["hold"] + 1, 0)
        info["ever"] = jp.maximum(
            info["ever"], (info["hold"] >= self._hold_needed).astype(jp.float32))
        # 次ステップの可動域。実機と同じで「直前に読んだ姿勢」から決まる。
        info["act_scale"] = self._act_scale(up)

        state.metrics["rec/upright"] = up
        state.metrics["rec/tilt"] = jp.linalg.norm(self._gravity(data)[:2])
        state.metrics["rec/height"] = h
        state.metrics["rec/feet"] = n_feet
        state.metrics["rec/ok"] = ok
        state.metrics["rec/trunk"] = (
            data.sensordata[self._trunk_contact_adr] > 0).astype(jp.float32)
        state.metrics["rec/success"] = info["ever"]
        state.metrics["rec/act_scale"] = info["act_scale"]
        # 正立していた時間の割合。歩行が生きているかの一次指標。
        state.metrics["rec/upright_frac"] = self._gate(up)
        # 報酬も同じ理由で無害化する。親は clip(-10, 10000) しているが、
        # NaN は clip を素通りする。
        return state.replace(
            reward=jp.nan_to_num(state.reward, nan=0.0, posinf=0.0, neginf=-10.0))

    def _status(self, data, info=None):
        """(水平度, 胴体高さ, 接地脚数, 復帰条件を満たしているか)。副作用なし。

        胴体高さは **足元の地面から** 測る (地形の上でも意味を保つ)。

        復帰版 (env/recover.py) の判定は「ほぼ静止していること」を含む。
        あちらは速度コマンドが常に 0 なので正しいが、**両立版では復帰した
        直後から歩けと指示されている**ことがあり、そのまま使うと「起き上がって
        指示どおり歩き出した」個体が永久に未復帰と数えられる (学習ログの
        「復帰 0%」はこれだった)。info を渡したときは、**歩けと指示されて
        いる間は静止条件を免除する**。報酬側は静止指令のときしか この項を
        使わない (recovered * still) ので、報酬の値はどちらでも変わらない。
        """
        rc = self._config.recover
        up = -self._gravity(data)[2]
        feet = data.site_xpos[self._feet_site_id]
        ground = jp.mean(self._ground_height(feet[..., :2]))
        h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS - ground
        contact = data.sensordata[self._feet_contact_adr] > 0
        n_feet = jp.sum(contact.astype(jp.float32))
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        w = self._sensor(data, consts.GYRO_SENSOR)
        settled = ((jp.linalg.norm(v) < rc.settle_lin)
                   & (jp.linalg.norm(w) < rc.settle_ang))
        if info is not None:
            settled = settled | (jp.linalg.norm(info["command"]) > 0.02)
        ok = ((up > rc.upright_cos)
              & (jp.abs(h - self._config.reward_config.base_height_target)
                 < rc.height_tol)
              & (n_feet >= self._feet_needed)
              & settled).astype(jp.float32)
        return up, h, n_feet, ok

    # ---------- 終了条件 ----------
    def _get_termination(self, data, info) -> jax.Array:
        """転倒では終了しない。打ち切るのは地形の外に出たときだけ。

        歩行版の「45度傾いたら終了」を残すと、転んだ瞬間にエピソードが終わって
        復帰を学べない。転倒を避ける圧力は罰ではなく「転ぶと歩行報酬を失う」
        ことで与える (upright が正立中ずっと +4/step 入るのがその本体)。
        """
        del info
        out = jp.max(jp.abs(data.xpos[self._torso_body_id, :2])) > \
            consts.TERRAIN_HALF - 1.0
        # **物理が壊れた env を捨てる**。接触が落ちて機体が床を抜けるなどで
        # qpos/qvel に NaN や Inf が出ることが稀にある (実測: 接触バッファ
        # 16/env で 256env x 1000step に 1 回。24/env でも 0 ではない)。
        # PPO はミニバッチに 1 つでも NaN が混ざると **全パラメータが 1 回の
        # 更新で NaN になる** (実測: run u3 は 26M step まで健全に伸びていて、
        # 次の eval で actor の 52,696 個すべてが NaN)。ここで done を立てれば
        # brax のラッパーが自動リセットするので、壊れた env は次のエピソードで
        # 復帰する。観測と報酬側の nan_to_num と合わせて 2 重に止める。
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))
        return out | bad

    # ---------- 報酬 ----------
    def _get_reward(self, data, action, info, done, first_contact, contact):
        rc = self._config.reward_config
        cc = self._config.recover
        uc = self._config.unified
        q = data.qpos[7:]

        up = -self._gravity(data)[2]
        a = self._gate(up)                  # 1 正立 / 0 転倒
        fallen = 1.0 - a

        # 歩行版の報酬項をそのまま作り、正立ぶんだけ効かせる。
        w = PupperJoystick._get_reward(
            self, data, action, info, done, first_contact, contact)
        out = {k: w[k] * a for k in WALK_GATED}
        # 正則化は両モードで必要。重みを a で補間する (REG_RATIO の注記)。
        for k, ratio in REG_RATIO.items():
            out[k] = w[k] * (a + fallen * ratio)
        out["dof_pos_limits"] = w["dof_pos_limits"]
        out["linkage_infeasible"] = w["linkage_infeasible"]
        out["termination"] = w["termination"]
        # トルク軟限界は割合そのものもモードで変える。復帰の大振りでは
        # 瞬間的な飽和は正常なので 0.8*stall まで許す。
        frac = a * rc.torque_soft_limit_frac \
            + fallen * uc.torque_soft_limit_frac_fallen
        over = jp.clip(jp.abs(data.actuator_force) - self._stall * frac, 0.0, None)
        out["torque_limit"] = jp.sum(jp.square(over)) * (
            a + fallen * uc.torque_limit_ratio)

        # --- 復帰系 (env/recover.py と同じ形。高さだけ地面相対にする) ---
        up_pos = jp.clip(up, 0.0, 1.0)
        feet = data.site_xpos[self._feet_site_id]
        ground = jp.mean(self._ground_height(feet[..., :2]))
        h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS - ground
        h_tgt = rc.base_height_target
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        gyro = self._sensor(data, consts.GYRO_SENSOR)
        trunk = (data.sensordata[self._trunk_contact_adr] > 0).astype(jp.float32)
        off = (1.0 - trunk) if cc.trunk_gate else 1.0
        gate = up_pos * off
        _, _, _, ok = self._status(data)
        # 復帰後の姿勢・静止を求める項は「速度コマンドが 0 のときだけ」。
        # 歩くように指示されているのに「基準姿勢で静止せよ」と言うと、
        # 歩行そのものが罰になる (rec7 が横倒しで止まったのと同じ形の壁)。
        still = (jp.linalg.norm(info["command"]) < 0.02).astype(jp.float32)

        out.update({
            # 傾き角に線形。cos をそのまま使うと仰向け (180度) で勾配が消え、
            # 最初の数度を揺すり出す動機が出ない。ここだけ角度で測る。
            # ゲートしない: 正立中の +4/step が「転ぶと失うもの」になる。
            "upright": 1.0 - jp.arccos(jp.clip(up, -1.0, 1.0)) / jp.pi,
            "height": fallen * gate * jp.exp(
                -jp.square(h - h_tgt) / cc.height_sigma),
            "height_low": fallen * up_pos * jp.clip(h_tgt - h, 0.0, None),
            "feet_down": fallen * gate * jp.mean(contact.astype(jp.float32)),
            "recovered": ok * still,
            "stand_pose": (up > cc.upright_cos) * still * jp.sum(
                jp.square(q - self._default_pose)),
            "settle": (up > 0.7) * still * (jp.sum(jp.square(v))
                                            + 0.1 * jp.sum(jp.square(gyro))),
            "trunk_contact": trunk,
            # 転倒しているぶんだけ立つ罰 (歩行版の termination 罰の代わり)。
            # 正立 (a=1) では厳密に 0 なので、歩行側の報酬バランスは
            # e40 のときと 1 ビットも変わらない。
            "fallen": fallen,
            "rock": jp.clip(up - info["best_up"], 0.0, None),
        })
        return out
