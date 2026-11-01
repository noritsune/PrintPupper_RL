"""PrintPupper 対角2脚での静止保持タスク (trot の中間姿勢を保つ)。

やること: 対角の2脚 (lf+rh または rf+lh) だけを接地させ、残り2脚を浮かせた
「トロットの中間姿勢」のまま、できるだけ長く倒れずに立っている。
遊脚が床に触れた瞬間にエピソード終了なので、エピソード長 = 保持時間になる。

なぜ難しいか:
  足先は球 (点接触) なので足首トルクが作れない。支持基底は対角2点を結ぶ
  線分だけで、その線まわりのモーメントは重心をずらす以外に止める手段がない。
  基準姿勢での重心は支持線から 3.4mm しか外れていない (ほぼ釣り合い) が、
  固定 PD で姿勢を保持するだけだと 0.48s で遊脚が接地する (実測)。
  使える手は
    - 接地脚の hip(外転) で足を横に振り、支持線そのものを動かす
      (脚長 0.16m なので 0.2rad の外転で足が 32mm 動く。重心オフセットの 10倍)
    - 遊脚を振って反作用トルクを作る
  の2つ。どちらも本物のバランス制御で、歩行方策の副産物では出てこない。

設計:
  観測は歩行版 (env/joystick.py) と同一の 38 次元。実機は IMU しか積んで
  いないので、ここでも関節エンコーダ無しで学習する。速度コマンドの 3 枠は
  [pair, 0, 0] に転用し、どちらの対角で立つかを方策に伝える
  (pair=+1: lf+rh 接地, -1: rf+lh 接地)。位相クロックは gait_freq=0 で
  止めてあるので定数 (1, 0) になる。

  action の基準姿勢だけを差し替える: 歩行版の
      目標角 = 4脚接地の基準姿勢 + action * 0.25
  に対し、ここでは
      目標角 = トロット中間姿勢 (遊脚を 0.40/-0.45 rad 引き上げ) + action * 0.25
  とする。基準を差し替えないと、学習初期の action≈0 が遊脚を床に落として
  しまい (遊脚の持ち上げ 0.45rad は action_scale 0.25 では届かない)、
  そもそも保持の試行が始まらない。action_scale は歩行版と同じ 0.25 のまま
  なので、実機側は基準姿勢の配列を差し替えるだけで済む。
"""
from typing import Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import constants as consts
from .joystick import PupperJoystick, default_config
from .linkage import infeasibility

# LEGS = [lf, rf, lh, rh]。トロットの対角ペア。
PAIR_A = jp.array([1.0, 0.0, 0.0, 1.0])     # lf + rh を接地
PAIR_B = jp.array([0.0, 1.0, 1.0, 0.0])     # rf + lh を接地


def balance_config() -> config_dict.ConfigDict:
    cfg = default_config()

    # 遊脚を基準姿勢から 0.40/-0.45 rad 引き上げると足先クリアランスは 42mm。
    # action_scale は歩行版と同じ 0.25 (基準姿勢のほうを差し替える)。
    cfg.gait_freq = [0.0, 0.0]          # 位相クロックは使わない (定数にする)

    # 外乱は既定で切る。まず「静かな床で立てる」ことを覚えさせ、必要なら
    # scripts/train.py --pert-kick で蹴りを入れて fine-tune する。
    cfg.pert_config.enable = False
    cfg.pert_config.velocity_kick = [0.0, 0.3]
    cfg.pert_config.kick_wait_times = [2.0, 5.0]

    cfg.balance = config_dict.create(
        lift=list(consts.BALANCE_LIFT),  # 遊脚の基準姿勢からの差分 [rad]
        pose_jitter=0.04,               # 初期関節角のばらつき [rad]
        clearance_target=0.035,         # 遊脚足先の目標クリアランス [m]
        randomize_pair=True,            # エピソードごとに対角ペアを選び直す
        init_vel=0.05,                  # 初期胴体速度のばらつき [m/s, rad/s]
        grace_steps=3,                  # リセット直後に遊脚接地を見逃すステップ数
    )

    # 重みの大前提: 「立っている間の1ステップ報酬が必ず正」であること。
    # 罰が生存報酬を上回ると、方策は「早く転んで罰の支払いを止める」のが
    # 最適になり、保持時間が縮む方向に学習が進む。実測で踏んだ罠:
    # 学習初期は行動ノイズ (|Δa|~1 -> 目標角 0.25rad ジャンプ -> kp=50 で
    # 12.5Nm) でサーボが常時飽和するので、torque_limit=-1.0 だと
    # -9.9/step となり生存 +2.5/step を軽く潰す。実際に 2M step で
    # 保持 0.36s -> 0.29s と悪化した。トルク罰はここでは弱く置き、
    # 保持を覚えてから重みを上げて fine-tune する
    # (飽和率そのものは progress の「トルク飽和」欄で常に見える)。
    cfg.reward_config = config_dict.create(
        scales=config_dict.create(
            # --- 目的そのもの ---
            survival=2.0,               # 生存 1 step ごと
            stance_contact=1.0,         # 接地脚がちゃんと着いている
            swing_clearance=-10.0,      # 遊脚が目標クリアランスより低い分
            # --- 姿勢 ---
            orientation=-3.0,
            base_height=-3.0,
            lin_vel_xy=-1.0,
            lin_vel_z=-1.0,
            ang_vel=-0.05,
            drift=-2.0,                 # スポーン位置からの水平移動
            feet_slip=-0.2,
            # --- 姿勢の形 (トロット中間姿勢らしさ) ---
            pose_swing=-0.5,            # 遊脚は持ち上げ姿勢を保つ
            pose_stance=-0.1,           # 接地脚は基準姿勢の近く (外転は許す)
            # --- 正則化 / ハードウェア保護 ---
            torques=-0.002,
            torque_limit=-0.1,
            action_rate=-0.02,
            energy=-0.001,
            dof_pos_limits=-2.0,
            linkage_infeasible=-5.0,
            termination=-5.0,
        ),
        base_height_target=consts.STAND_HEIGHT,
        torque_soft_limit_frac=0.5,
    )
    return cfg


class PupperBalance(PupperJoystick):
    """対角2脚での静止保持。観測次元は歩行版と同一 (actor 38)。"""

    def __init__(
        self,
        config: config_dict.ConfigDict = balance_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def _post_init(self) -> None:
        super()._post_init()
        bc = self._config.balance
        self._lift = jp.array(np.tile(np.asarray(bc.lift, np.float32), 4))  # (12,)
        self._grace = int(bc.grace_steps)

    # ---------- reset ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # 歩行版の reset をそのまま使い (info の作り方・乱数消費を共有)、
        # そのあと遊脚だけ持ち上げた姿勢に差し替える。
        state = super().reset(rng)
        info = state.info
        bc = self._config.balance

        info["rng"], k_pair, k_jit, k_vel = jax.random.split(info["rng"], 4)
        pick_a = (jax.random.bernoulli(k_pair) if bc.randomize_pair
                  else jp.array(True))
        stance = jp.where(pick_a, PAIR_A, PAIR_B)
        pair = jp.where(pick_a, 1.0, -1.0)
        swing = 1.0 - stance

        # トロット中間姿勢 = 基準姿勢 + 遊脚だけ持ち上げ。これが action の
        # 基準 (_motor_targets) であり、pose_swing 報酬の目標でもある。
        # 対角ペアだけで決まる決定的な値なので、実機側も同じ配列を持てばよい。
        q_ref = self._default_pose + (
            self._lift.reshape(4, 3) * swing[:, None]).ravel()
        # 初期姿勢だけを ±pose_jitter だけ崩す (PD がすぐ q_ref へ引き戻す)。
        qpos = state.data.qpos.at[7:].set(q_ref + jax.random.uniform(
            k_jit, (12,), minval=-bc.pose_jitter, maxval=bc.pose_jitter))
        qvel = state.data.qvel.at[0:6].set(
            jax.random.uniform(k_vel, (6,), minval=-bc.init_vel, maxval=bc.init_vel))
        data = state.data.replace(qpos=qpos, qvel=qvel, ctrl=q_ref)
        data = mjx.forward(self.mjx_model, data)

        # 胴体高さは keyframe の値 (4脚接地) のままでは合わない。関節角の
        # ばらつきぶん接地脚が浮いたりめり込んだりしたまま始まると、最初の
        # 数ステップで勝手に倒れ込む。接地脚の低いほうがちょうど床に触れる
        # 高さへ置き直す。1mm だけ浮かせて初期の押し込みを避ける。
        feet_z = data.site_xpos[self._feet_site_id][:, 2] - consts.FOOT_RADIUS
        drop = jp.min(jp.where(stance > 0.5, feet_z, jp.inf)) - 0.001
        qpos = qpos.at[2].add(-drop)
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos))

        info["stance"] = stance
        info["q_ref"] = q_ref
        info["spawn_xy"] = qpos[:2]
        info["hold_steps"] = jp.zeros((), jp.int32)
        # 速度コマンドの枠を「どちらの対角で立つか」に転用する。
        info["command"] = jp.array([pair, 0.0, 0.0])
        info["phase"] = jp.zeros(())
        info["gait_freq"] = jp.zeros(())

        metrics = dict(state.metrics)
        for k in ("bal/swing_z", "bal/tilt", "bal/drift", "bal/stance",
                  "bal/hold"):
            metrics[k] = jp.zeros(())

        obs = self._obs_reset(data, info)
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    def _motor_targets(self, applied: jax.Array, info) -> jax.Array:
        return info["q_ref"] + applied * self._config.action_scale

    # ---------- step ----------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # termination の猶予判定に使うので、親の step に入る前に進めておく。
        state.info["hold_steps"] = state.info["hold_steps"] + 1
        state = super().step(state, action)
        info, data = state.info, state.data

        stance = info["stance"]
        feet_xpos = data.site_xpos[self._feet_site_id]
        foot_z = (feet_xpos[:, 2] - consts.FOOT_RADIUS
                  - self._ground_height(feet_xpos[:, :2]))
        contact = (data.sensordata[self._feet_contact_adr] > 0).astype(jp.float32)
        gravity = self._gravity(data)

        state.metrics["bal/swing_z"] = (
            jp.sum(foot_z * (1.0 - stance)) / jp.sum(1.0 - stance))
        state.metrics["bal/tilt"] = jp.linalg.norm(gravity[:2])
        state.metrics["bal/drift"] = jp.linalg.norm(
            data.xpos[self._torso_body_id, :2] - info["spawn_xy"])
        state.metrics["bal/stance"] = jp.sum(contact * stance) / jp.sum(stance)
        state.metrics["bal/hold"] = 1.0 - state.done
        # 親が入れる歩行用の指標は意味が変わるので、この課題の量に差し替える。
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        state.metrics["track/speed"] = jp.linalg.norm(v[:2])
        state.metrics["track/cmd_speed"] = jp.zeros(())
        state.metrics["track/lin_err"] = state.metrics["bal/drift"]
        state.metrics["track/ang_err"] = jp.abs(
            self._sensor(data, consts.GYRO_SENSOR)[2])

        # done なら次エピソードのために猶予カウンタを戻す (AutoReset は
        # data/obs しか巻き戻さず info はそのまま流れてくる)。
        state.info["hold_steps"] = jp.where(
            state.done > 0, 0, state.info["hold_steps"])
        return state

    # ---------- 終了条件 ----------
    def _get_termination(self, data, info) -> jax.Array:
        base = super()._get_termination(data, info)
        contact = data.sensordata[self._feet_contact_adr] > 0
        swing_touch = jp.any(contact & (info["stance"] < 0.5))
        grace = info["hold_steps"] <= self._grace
        return base | (swing_touch & ~grace)

    # ---------- コマンド ----------
    def sample_command(self, rng: jax.Array, prev: jax.Array) -> jax.Array:
        # 対角ペアの指定はエピソード内で変えない。
        del rng
        return prev

    # ---------- 報酬 ----------
    def _get_reward(self, data, action, info, done, first_contact, contact):
        rc = self._config.reward_config
        bc = self._config.balance
        stance = info["stance"]
        swing = 1.0 - stance
        stance_j = jp.repeat(stance, 3)          # (12,) 関節ごとのマスク
        swing_j = jp.repeat(swing, 3)

        q = data.qpos[7:]
        gravity = self._gravity(data)
        gyro = self._sensor(data, consts.GYRO_SENSOR)
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        contact_f = contact.astype(jp.float32)

        feet_xpos = data.site_xpos[self._feet_site_id]
        ground = self._ground_height(feet_xpos[:, :2])
        foot_z = feet_xpos[:, 2] - consts.FOOT_RADIUS - ground
        feet_vel = data.sensordata[
            (self._feet_linvel_adr[:, None] + np.arange(3)[None])
        ]

        base_h = (data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS
                  - jp.mean(ground))
        over = jp.clip(jp.abs(data.actuator_force)
                       - self._stall * rc.torque_soft_limit_frac, 0.0, None)
        out_of_limits = jp.sum(
            jp.clip(self._soft_lowers - q, 0.0, None)
            + jp.clip(q - self._soft_uppers, 0.0, None)
        )

        return {
            "survival": jp.ones(()),
            "stance_contact": jp.sum(contact_f * stance) / jp.sum(stance),
            # 目標より低い分だけを線形に罰する。上げすぎは罰さない
            # (上げるほど良いわけでもないので pose_swing が形を決める)。
            "swing_clearance": jp.sum(
                jp.clip(bc.clearance_target - foot_z, 0.0, None) * swing),
            "orientation": jp.sum(jp.square(gravity[:2])),
            "base_height": jp.square(base_h - rc.base_height_target),
            "lin_vel_xy": jp.sum(jp.square(v[:2])),
            "lin_vel_z": jp.square(v[2]),
            "ang_vel": jp.sum(jp.square(gyro)),
            "drift": jp.sum(jp.square(
                data.xpos[self._torso_body_id, :2] - info["spawn_xy"])),
            "feet_slip": jp.sum(jp.square(feet_vel[:, :2]) * contact[:, None]),
            "pose_swing": jp.sum(jp.square(q - info["q_ref"]) * swing_j),
            "pose_stance": jp.sum(jp.square(q - self._default_pose) * stance_j),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "torque_limit": jp.sum(jp.square(over)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            "dof_pos_limits": out_of_limits,
            "linkage_infeasible": infeasibility(q),
            "termination": done.astype(jp.float32),
        }
