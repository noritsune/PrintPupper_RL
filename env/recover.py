"""PrintPupper 転倒復帰タスク (ひっくり返った状態から自力で立ち上がる)。

やること: 仰向け / 横倒し / 正立 のいずれかで床に置かれた状態から始め、
4脚接地の基準姿勢まで自力で復帰し、そのまま静止して待つ。復帰後は歩行方策
(e40) にそのまま引き渡せる姿勢・高さで止まることを要求する。

**歩行方策の可動域では復帰できない** (この課題で最初に当たる壁):
  歩行版の関節指令は「4脚接地の基準姿勢 + action * 0.25 rad」。方策の出力は
  tanh で [-1,1] に収まるので、可動域は厳密に「基準姿勢 ± action_scale」になる。
  仰向けの機体は胴体箱の上面 (機体フレーム z = +57mm) で床に乗り、ロール角 α
  のときの接地稜は (y=+64.5mm, z=+57mm)。機体上の点 (y, z) が床に届く条件は

      z + y*tanα >= 57 + 64.5*tanα   [mm]

  なので、脚を横に大きく開くほど (y を稼ぐほど) 浅いロールで床に届く。
  4節リンクで実現可能な姿勢だけに絞って実測した「足先を床に着けるのに必要な
  ロール角」:

      action_scale 0.25 -> 73度 (足先の y 到達 131mm)   ... 事実上不可能
      action_scale 0.80 -> 48度 (228mm)
      action_scale 1.00 -> 37度 (254mm)
      action_scale 1.20 -> 27度 (265mm)
      action_scale 1.60 ->  4度 (270mm)  <- ここを使う

  平らに寝た状態 (ロール 0度) から足先が届くかだけを見ると、リンク実現可能な
  範囲での足先の最高到達点は機体フレーム z = +49mm で、床の +57mm に 8mm
  足りない。つまり「寝たまま足で床を押す」は使えず、まず脚の反作用で数度
  揺すってから足を着く必要がある。action_scale 1.6 なら必要なロールが 4度まで
  下がるので、これは脚を振れば必ず作れる。1.6 は「全関節の可動域を丸ごと
  使える」値でもある (hip ±1.5708 を含む)。action=0 が基準姿勢のままなので、
  復帰後の静止と歩行方策への引き渡しはそのまま成立する。
  実機側は env/deploy.py の ObsBuilder(action_scale=RECOVER_ACTION_SCALE)。

  参考: 開ループの軌道探索 (CEM) でも、全可動域を使えれば 3秒で起き上がる
  軌道が見つかる (scripts/ ではなく作業用スクリプトで確認)。ただし見つかった
  軌道は膝が 4節リンクの解けない領域に入っていたので、**リンクの実現可能性を
  守ったままの復帰動作は自明ではない**。linkage_infeasible の罰は必ず残すこと。

設計:
  - 観測は歩行版と同一の 38 次元。実機は IMU しか積んでいないので姿勢の
    手がかりは重力方向 3 次元だけ。速度コマンド枠は [0,0,0] 固定、位相
    クロックは gait_freq=0 で止める (balance 版と同じ扱い)。
  - **途中終了しない**。歩行版は「45度傾いたら終了」だが、この課題は傾いて
    いる状態が入口なのでその判定を外す。エピソード長は常に一定で、成績は
    「復帰できた割合」と「復帰までの時間」で測る (scripts/recover_test.py)。
  - 報酬の主軸は upvector の z (= 胴体の水平度)。仰向け -1 -> 正立 +1 の
    連続量なので、転がる途中にも勾配がある。高さ・接地・静止の各項は
    「正立しているぶんだけ」効くようゲートしてあり (up_pos)、仰向けのまま
    胴体を持ち上げて点を稼ぐ抜け道を塞いでいる。
  - 4節リンクの実現可能性 (linkage_infeasible) は歩行版と同じ重みで残す。
    復帰動作は関節を端まで使うので、ここを外すと実機で出せない姿勢に
    逃げる。実測ではリンク実現可能な範囲だけでも足先は機体フレーム
    z = +110mm まで届く (床の 53mm 下) ので、性能上の制約にはならない。
"""
from typing import Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as pg_math
from mujoco_playground._src import mjx_env

from . import constants as consts
from .joystick import PupperJoystick, default_config
from .linkage import infeasibility

def recover_config() -> config_dict.ConfigDict:
    cfg = default_config()

    # 復帰は数秒で終わる。残りは「立ったまま待てるか」を見る時間。
    cfg.episode_length = 400            # 8 s
    # 上の docstring の実測より、1.2 では 27度のロールを作らないと足が床に
    # 着かない。1.6 なら 4度で足り、かつ全関節の可動域を丸ごと使える。
    cfg.action_scale = 1.6
    cfg.gait_freq = [0.0, 0.0]          # 位相クロックは使わない (定数にする)
    cfg.pert_config.enable = False
    # 仰向けでは胴体箱と床の接触が最大4点、これに足4点が乗る。歩行版の
    # 8点/env では溢れて機体が床を抜ける。
    cfg.naconmax = 12 * 4096

    cfg.recover = config_dict.create(
        # 初期姿勢の内訳。残り (1 - p_flip - p_side) は正立で置く。
        # 正立を混ぜるのは「もう立っているなら何もせず立ち続ける」を
        # 同じ方策に入れておくため (実機では姿勢を選んで起動できない)。
        # rec3 まで 仰向け 0.60 / 横倒し 0.25 / 正立 0.15。仰向けからの成功率が
        # 0% で頭打ちになった原因は「起き上がれない」ことではなく (転がって
        # 水平になるのは 1秒で確実にできる)、**そこから 4脚で立ち上がれない**
        # ことだった。伏せの姿勢 (胴体高 105mm / 目標 163mm) は「起き上がった
        # 後」にしか現れないので、1エピソードあたりの練習量が少ないうえに、
        # そこへ至る行動と絡んで信用割当が難しい。初期姿勢に直接入れる。
        # rec4 で伏せを 0.20 入れたところ「伏せ -> 立つ」は 86% になり、全体の
        # 成功率が 17.2% -> 33.6% に上がった。ただし仰向けからの成功は 0% の
        # まま。連鎖の後半が解けたので、rec5 では仰向けの練習量を戻す。
        p_flip=0.65,                    # 仰向け (roll ≈ 180度)
        p_side=0.15,                    # 横倒し (roll ≈ ±90度)
        p_prone=0.15,                   # 伏せ (正立だが脚を前に投げ出した姿勢)
        # 伏せ姿勢の関節角 [hip, upper_leg, lower_leg] の基準姿勢からの差分。
        # rec3 の方策が実際に収束した姿勢 (前脚を前へ投げ出し、後脚を畳む) を
        # 模した値。ここから立ち上がるのがこの課題の後半そのもの。
        prone_delta=[0.0, -0.7, -0.6],
        roll_jitter=0.6,                # 各モードの roll のばらつき [rad]
        pitch_jitter=0.4,
        pose_jitter=0.35,               # 初期関節角のばらつき [rad]
        clearance=0.002,                # 置いた瞬間の浮かせ量 [m]
        init_vel=0.10,                  # 初期並進速度のばらつき [m/s]
        init_avel=0.50,                 # 初期角速度のばらつき [rad/s]
        # --- 「復帰した」の判定 ---
        upright_cos=0.85,               # upvector z がこれ以上 (= 傾き32度以内)
        height_tol=0.030,               # 胴体高さの許容誤差 [m]
        feet_needed=3,                  # 接地している足の本数
        settle_lin=0.10,                # 胴体並進速度 [m/s]
        settle_ang=1.00,                # 胴体角速度 [rad/s]
        hold_steps=25,                  # 連続 0.5s 満たせば「成功」
        # 高さ報酬の幅。**ここを狭くすると立ち上がらない**。run rec1 は
        # 0.0008 で回して仰向けからの成功 0% に終わった。原因は報酬の形で、
        # exp(-e^2/0.0008) は誤差 70mm で 0.001 = 勾配が完全に消える。
        # 方策は「水平にはなったが胴体高 91mm (目標 163mm) のまま」で収束し、
        # height 項はエピソード合計 +31 / 最大 1200 しか取れていなかった
        # (upright が +1684 で支配的)。0.004 なら 90mm でも 0.27 出る。
        height_sigma=0.004,
        # **胴体が床に着いている間は height / feet_down を無効にする**。
        # rec1-rec6 が仰向けから抜けられなかった真因はこれだった。方策は
        # 胴体の前端を床に着けたまま前脚だけで機体を持ち上げる「逆立ち」に
        # 収束する。この姿勢は
        #   水平度 +0.54 / 胴体高 132mm (目標 163mm の 81%) / 前脚2本だけ接地
        # なので height (+1.3/step) と feet_down (+1.1/step) をほぼ満額
        # 取れてしまい、しかも **そこから動くと必ず水平度が下がる**
        # (実測: action=0 に戻すと +0.54 -> +0.21、前脚を畳むと -0.55)。
        # 報酬の局所最大に自分で入り込んで出られない状態だった。
        # 胴体が床から離れているときだけ高さと接地を数えれば、この姿勢の
        # 価値は +4.5/step から -1.0/step に落ちる (正立は +15/step)。
        trunk_gate=True,
        # height / height_low が測る「高さ」の定義。
        #   "origin"    胴体フレーム原点の z (rec1-rec7)
        #   "clearance" **胴体箱の最下点の床からの高さ**
        # origin は機体を傾けても稼げるのが弱点で、逆立ち (原点 132mm =
        # 目標の 81%) がほぼ満額取れてしまう。clearance ならこの姿勢は 0mm
        # なので、二値の trunk_gate を使わずに同じことが連続量でできる。
        # 4脚接地の基準姿勢では箱の底は床上 159.6mm (= 原点 174.6 - 15.0)。
        height_mode="origin",
        # 詰まり状態からの初期化 (scripts/make_stuck_states.py が作る .npz)。
        # 「その姿勢でしか練習できない動作」を初期姿勢に直接入れる手は
        # rec4 の「伏せ」で既に効いている。今回は前の run が実際に詰まった
        # 状態そのものを記録して混ぜる。
        p_stuck=0.0,
    )

    # 重みの大前提は balance 版と同じ: 立っている間の1ステップ報酬が必ず正で
    # あること。ただしこの課題は途中終了が無いので「早く転んで罰を止める」
    # 縮退は起こらない。むしろ危険なのは逆で、正則化を強くすると
    # 「動かずに静かに仰向けでいる」のが最適になる。torques / action_rate は
    # 歩行版より弱く置き、復帰に必要な大振りの動きを殺さないようにする。
    # 重みの実測 (4M step の smoke run, 1ステップあたり *dt 前):
    #   torque_limit=-0.1 / soft_frac=0.5 で -2.40/step。目的の upright
    #   (+0.33/step) の 7倍あり、方策は 4M step で早くも「動かない」方向に
    #   倒れた (upright 0.33 -> 0.17)。action_scale 1.6 では PD 目標が現在角
    #   から最大 1.6rad 離れるので kp=50 なら要求トルクは 80Nm 相当、
    #   大きく動く限りサーボは必ず飽和する。**復帰動作で瞬間的に飽和するのは
    #   正常**で、実機で問題なのは持続的なストールのほうなので、
    #   軟限界を 0.8*stall まで上げ、重みも 1/10 にする。
    cfg.reward_config = config_dict.create(
        scales=config_dict.create(
            # --- 目的そのもの ---
            # run rec2 まで 6.0。エピソード合計 +1637 / 全体 2052 と支配的で、
            # 「低いが水平」(胴体高 105mm) で収束した。高さと姿勢に配分を回す。
            upright=4.0,        # 傾き角に線形 (仰向け 0 / 横倒し 0.5 / 正立 1)
            height=3.0,         # 立ち上がり切った高さ (正立ぶんだけ有効)
            # 高さの「不足ぶん」を線形に罰する。exp 型の height だけだと
            # 目標から離れた場所で勾配が消えるので、遠くから引き上げる力を
            # 別に用意する。こちらは常に一定の勾配がある。
            height_low=-25.0,
            # rec1-rec6 が止まる姿勢は「右前脚を畳んだ3脚立ち」だと見ていたが、
            # 実際は **胴体の前端を床に着けた逆立ち** だった (接触 geom を
            # 直接見ると floor/lf_foot_col, floor/rf_foot_col, floor/trunk_col
            # の3点、後脚の足先は床から 85-140mm 浮いている)。前脚2本の接地で
            # この項を半額 (+1.1/step) 取れるのが居座りを支えていたので、
            # trunk_gate で胴体接地中は 0 にする。
            feet_down=4.0,      # 足で立っている (同上)
            recovered=4.0,      # 復帰条件を全部満たしている
            # 「これまでの最高到達点を更新したぶん」だけ加点する (既定 0)。
            # 総和が「最終到達 - 初期」に telescoping するので、上下動を
            # 繰り返して無限に稼ぐことはできない (terrain.py の climb と同じ形)。
            # 仰向けから数度揺するところに報酬が全く付かないときの保険。
            # 使うときは --scale rock=2.0。
            rock=0.0,
            # --- 復帰後の形 ---
            # rec2 の失敗: 起き上がりは 1秒で成功するのに、前脚を投げ出した
            # 「伏せ」(胴体高 105mm / 目標 163mm) で止まる。脚が基準姿勢から
            # 外れていることを罰するのはこの項だけなので、-0.5 では弱い。
            # **rec7 が横倒しで止まった原因はこの項だった**。ゲートが up_pos
            # だったので、横倒し (水平度 0) では罰が 0 になり、方策は脚を
            # めちゃくちゃに広げたまま (逸脱 10.2 rad^2) そこに居座れる。
            # そこから起き上がろうとすると up_pos が開くぶんだけ罰が立ち上がり、
            #   水平度 +0.2 で -16/step, +0.5 で -41/step, +0.8 で -66/step
            # に対し upright の得はせいぜい +1.6/step。**起き上がる動作そのものが
            # 罰になる壁**だった (rec6 で -3.0 -> -8.0 に上げたのが裏目)。
            # この項の目的は「歩行方策に渡せる姿勢で止まる」ことだけなので、
            # **ほぼ正立したときにだけ効かせる** (ゲートを up > upright_cos に
            # する)。途中経過の姿勢には一切口を出さない。
            stand_pose=-3.0,    # 基準姿勢からのずれ (歩行方策への引き渡し用)
            settle=-0.3,        # 復帰後に暴れない (ok ゲート: 下の注記)
            # rec6 まで -0.3。逆立ちで止まる局所最大の価値が +4.5/step も
            # あったので、そこに居座るコストとして軽すぎた。trunk_gate と
            # 合わせて「胴体を床に着けたままの姿勢は 1ステップ報酬が必ず負」
            # にする。仰向けの間は避けようのない支払い (-2.0/step x 0.5s)
            # だが、途中終了が無いので「早く転んで支払いを止める」縮退は
            # 起きない。
            trunk_contact=-2.0, # 胴体が床に着いている
            # --- 正則化 / ハードウェア保護 ---
            # 合計で -0.2/step 程度に収める。ここを厚くすると「仰向けのまま
            # 動かない」が最適になる (balance 版と同じ罠だが、あちらは
            # 「早く転ぶ」、こちらは「何もしない」の形で出る)。
            torques=-0.0005,
            torque_limit=-0.01,
            action_rate=-0.01,
            energy=-0.0005,
            dof_pos_limits=-2.0,
            linkage_infeasible=-5.0,
        ),
        base_height_target=consts.STAND_HEIGHT,
        # 復帰は瞬間的に全力を出す動作なので、歩行版の 0.5 より緩める。
        torque_soft_limit_frac=0.8,
    )
    return cfg


class PupperRecover(PupperJoystick):
    """転倒復帰。観測次元は歩行版と同一 (actor 38)。"""

    # 脚にも衝突形状を付けた XML を使う (env/constants.py の RECOVER_XML)。
    # 歩行版の XML は脚が床をすり抜けるので、脚で床を押す動作が学べない。
    XML = consts.RECOVER_XML

    def __init__(
        self,
        config: config_dict.ConfigDict = recover_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def _post_init(self) -> None:
        super()._post_init()
        m = self._mj_model
        self._trunk_geom_id = m.geom("trunk_col").id
        self._trunk_size = jp.array(m.geom_size[self._trunk_geom_id])
        # 箱の8隅 (ローカル)。置き直しのときに床へのめり込みを見る。
        self._box_corners = jp.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=jp.float32)
        rc = self._config.recover
        self._prone_delta = jp.array(
            np.tile(np.asarray(rc.prone_delta, np.float32), 4))
        self._hold_needed = int(rc.hold_steps)
        self._feet_needed = int(rc.feet_needed)
        self._stuck = None          # load_stuck_states() で入る
        # 胴体箱の最下点を測るための基準 (height_mode="clearance")。
        # 4脚接地の基準姿勢では原点 z = STAND_HEIGHT + FOOT_RADIUS、
        # 箱の底は原点から (pos_z - size_z) だけ下。
        self._trunk_bottom = float(
            m.geom_pos[self._trunk_geom_id][2] - m.geom_size[self._trunk_geom_id][2])
        self._clear_target = (consts.STAND_HEIGHT + consts.FOOT_RADIUS
                              + self._trunk_bottom)

    # ---------- 詰まり状態からの初期化 ----------
    def load_stuck_states(self, path) -> int:
        """前の run が実際に詰まった状態 (.npz) を初期姿勢の候補に加える。

        scripts/make_stuck_states.py が qpos / qvel / act を記録する。
        「その姿勢でしか練習できない動作」を初期姿勢に直接入れる手であり、
        rec4 の「伏せ」で既に効いている。ここで大事なのは **行動履歴も
        一緒に復元する**こと。実機に関節エンコーダが無いので観測の自己
        受容感覚は行動履歴だけで、ここを 0 にすると「脚がどこにあるか」の
        手がかりが消えた別の状態になってしまう。
        """
        d = np.load(path)
        self._stuck = (jp.array(d["qpos"], jp.float32),
                       jp.array(d["qvel"], jp.float32),
                       jp.array(d["act"], jp.float32))
        return int(d["qpos"].shape[0])

    # ---------- reset ----------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        # 歩行版の reset で info を作らせてから、姿勢だけ差し替える
        # (乱数の消費の仕方と info のキーを共有しておく)。
        state = super().reset(rng)
        info = state.info
        rc = self._config.recover

        info["rng"], k_mode, k_roll, k_pitch, k_yaw, k_q, k_v = jax.random.split(
            info["rng"], 7)

        # --- 向き ---
        u = jax.random.uniform(k_mode)
        # 仰向け: roll=pi, 横倒し: roll=±pi/2, 伏せ/正立: roll=0
        side_sign = jp.where(jax.random.bernoulli(k_yaw), 1.0, -1.0)
        roll0 = jp.where(u < rc.p_flip, jp.pi,
                         jp.where(u < rc.p_flip + rc.p_side,
                                  side_sign * jp.pi / 2, 0.0))
        prone = (u >= rc.p_flip + rc.p_side) & (
            u < rc.p_flip + rc.p_side + rc.p_prone)
        roll = roll0 + jax.random.uniform(
            k_roll, (), minval=-rc.roll_jitter, maxval=rc.roll_jitter)
        pitch = jax.random.uniform(
            k_pitch, (), minval=-rc.pitch_jitter, maxval=rc.pitch_jitter)
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        quat = pg_math.quat_mul(
            pg_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw),
            pg_math.quat_mul(
                pg_math.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), pitch),
                pg_math.axis_angle_to_quat(jp.array([1.0, 0.0, 0.0]), roll)))

        # --- 関節 ---
        # 実機が転んだ直後はサーボが最後の指令 (基準姿勢) を保持している。
        # そこからのばらつきを与える。
        q0 = jp.clip(
            self._default_pose
            + jp.where(prone, self._prone_delta, 0.0)
            + jax.random.uniform(
                k_q, (12,), minval=-rc.pose_jitter, maxval=rc.pose_jitter),
            self._soft_lowers, self._soft_uppers)

        qpos = state.data.qpos
        qpos = qpos.at[0:2].set(0.0).at[2].set(0.4)   # 一旦浮かせて FK を取る
        qpos = qpos.at[3:7].set(quat).at[7:].set(q0)
        kv1, kv2 = jax.random.split(k_v)
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[0:3].set(jax.random.uniform(
            kv1, (3,), minval=-rc.init_vel, maxval=rc.init_vel))
        qvel = qvel.at[3:6].set(jax.random.uniform(
            kv2, (3,), minval=-rc.init_avel, maxval=rc.init_avel))
        data = mjx.forward(self.mjx_model, state.data.replace(
            qpos=qpos, qvel=qvel, ctrl=q0))

        # --- 床の上に置き直す ---
        # 接触する可能性があるのは足球4個と胴体箱だけ (脚リンクに衝突形状は
        # 無い)。その最下点がちょうど床に触れる高さまで落とす。どの向きでも
        # 効くので、仰向け・横倒し・正立を同じコードで扱える。
        feet_z = data.site_xpos[self._feet_site_id][:, 2] - consts.FOOT_RADIUS
        gpos = data.geom_xpos[self._trunk_geom_id]
        gmat = data.geom_xmat[self._trunk_geom_id].reshape(3, 3)
        corners_z = gpos[2] + (self._box_corners * self._trunk_size) @ gmat[2]
        drop = jp.minimum(jp.min(feet_z), jp.min(corners_z)) - rc.clearance
        qpos = qpos.at[2].add(-drop)
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos))

        # --- 詰まり状態への差し替え (p_stuck の確率) ---
        if self._stuck is not None and float(rc.p_stuck) > 0.0:
            sq, sv, sa = self._stuck
            info["rng"], k_i, k_u = jax.random.split(info["rng"], 3)
            i = jax.random.randint(k_i, (), 0, sq.shape[0])
            use = (jax.random.uniform(k_u) < rc.p_stuck).astype(jp.float32)
            qpos = qpos * (1.0 - use) + sq[i] * use
            qvel = qvel * (1.0 - use) + sv[i] * use
            a0 = sa[i] * use
            data = mjx.forward(self.mjx_model, data.replace(
                qpos=qpos, qvel=qvel,
                ctrl=self._default_pose + a0 * self._config.action_scale))
            # 行動履歴も復元する (観測の自己受容感覚はこれしかない)。
            info["last_act"] = a0
            info["last_last_act"] = a0
            info["action_buf"] = jp.tile(a0, (info["action_buf"].shape[0], 1))

        info["command"] = jp.zeros(3)      # 速度コマンドは使わない
        info["phase"] = jp.zeros(())
        info["gait_freq"] = jp.zeros(())
        info["hold"] = jp.zeros((), jp.int32)     # 連続で復帰条件を満たした数
        info["best_up"] = -self._gravity(data)[2]  # これまでの水平度の最高値
        info["ever"] = jp.zeros(())               # 一度でも成功したか

        metrics = dict(state.metrics)
        for k in ("rec/upright", "rec/height", "rec/feet", "rec/ok",
                  "rec/success", "rec/tilt", "rec/trunk"):
            metrics[k] = jp.zeros(())

        obs = self._obs_reset(data, info)
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    # ---------- step ----------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        data, info = state.data, state.info

        up, h, n_feet, ok = self._status(data)
        # rock 報酬はここで更新する (_get_reward は更新前の値を見る)。
        info["best_up"] = jp.maximum(info["best_up"], up)
        # hold / ever の更新はここ 1 か所だけ。_get_reward からも同じ判定を
        # 使うが、あちらは副作用を持たない (二重カウントを避ける)。
        info["hold"] = jp.where(ok > 0, info["hold"] + 1, 0)
        info["ever"] = jp.maximum(
            info["ever"], (info["hold"] >= self._hold_needed).astype(jp.float32))
        state.metrics["rec/upright"] = up
        state.metrics["rec/tilt"] = jp.linalg.norm(self._gravity(data)[:2])
        state.metrics["rec/height"] = h
        state.metrics["rec/feet"] = n_feet
        state.metrics["rec/ok"] = ok
        # 胴体が床に着いている時間の割合。rec1-rec6 が抜けられなかった
        # 「逆立ち」はこれが 100% のまま止まるので、報酬より先にここを見る。
        state.metrics["rec/trunk"] = (
            data.sensordata[self._trunk_contact_adr] > 0).astype(jp.float32)
        state.metrics["rec/success"] = info["ever"]
        # 親が入れる歩行用の指標は意味が変わるので差し替える。
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        state.metrics["track/speed"] = jp.linalg.norm(v[:2])
        state.metrics["track/cmd_speed"] = jp.zeros(())
        state.metrics["track/lin_err"] = jp.zeros(())
        state.metrics["track/ang_err"] = jp.zeros(())
        return state

    def _status(self, data):
        """(水平度, 胴体高さ, 接地脚数, 復帰条件を満たしているか) を返す。副作用なし。"""
        rc = self._config.recover
        up = -self._gravity(data)[2]                       # +1 正立 / -1 仰向け
        h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS
        contact = data.sensordata[self._feet_contact_adr] > 0
        n_feet = jp.sum(contact.astype(jp.float32))
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        w = self._sensor(data, consts.GYRO_SENSOR)
        ok = ((up > rc.upright_cos)
              & (jp.abs(h - self._config.reward_config.base_height_target)
                 < rc.height_tol)
              & (n_feet >= self._feet_needed)
              & (jp.linalg.norm(v) < rc.settle_lin)
              & (jp.linalg.norm(w) < rc.settle_ang)).astype(jp.float32)
        return up, h, n_feet, ok

    # ---------- 終了条件 ----------
    def _get_termination(self, data, info) -> jax.Array:
        # この課題は「傾いている」ことが入口なので、歩行版の転倒判定
        # (45度 / 胴体接触) をそのまま使うと最初の1ステップで終了する。
        # 途中終了は一切させず、エピソード長を固定する。
        del data, info
        return jp.zeros((), dtype=bool)

    # ---------- コマンド ----------
    def sample_command(self, rng: jax.Array, prev: jax.Array) -> jax.Array:
        del rng, prev
        return jp.zeros(3)

    # ---------- 報酬 ----------
    def _get_reward(self, data, action, info, done, first_contact, contact):
        rc = self._config.reward_config
        cc = self._config.recover
        q = data.qpos[7:]

        up = -self._gravity(data)[2]
        # 正立ぶんだけ効かせるゲート。仰向けのまま胴体を持ち上げて高さ報酬を
        # 稼ぐ抜け道を塞ぐ。連続なので転がる途中でも勾配がある。
        up_pos = jp.clip(up, 0.0, 1.0)

        h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        w = self._sensor(data, consts.GYRO_SENSOR)
        contact_f = contact.astype(jp.float32)
        # height / height_low が見る高さ。rec/height (成功判定) は常に原点で
        # 測る (判定の定義を run ごとに動かさないため)。ここは報酬だけ。
        if cc.height_mode == "clearance":
            gpos = data.geom_xpos[self._trunk_geom_id]
            gmat = data.geom_xmat[self._trunk_geom_id].reshape(3, 3)
            h_rew = jp.min(
                gpos[2] + (self._box_corners * self._trunk_size) @ gmat[2])
            h_tgt = self._clear_target
        else:
            h_rew, h_tgt = h, rc.base_height_target
        _, _, _, ok = self._status(data)

        over = jp.clip(jp.abs(data.actuator_force)
                       - self._stall * rc.torque_soft_limit_frac, 0.0, None)
        out_of_limits = jp.sum(
            jp.clip(self._soft_lowers - q, 0.0, None)
            + jp.clip(q - self._soft_uppers, 0.0, None))
        trunk = (data.sensordata[self._trunk_contact_adr] > 0).astype(jp.float32)
        # 胴体が床から離れているか。height / feet_down はこれで追加ゲートする
        # (理由は recover_config の trunk_gate の注記)。**height_low は
        # ゲートしない**: あれは「低いところから引き上げる」罰なので、
        # 胴体が床に着いているときこそ効いていないといけない。
        off = (1.0 - trunk) if cc.trunk_gate else 1.0
        gate = up_pos * off

        return {
            # 傾き角に線形にする。cos(傾き) をそのまま使うと、仰向け
            # (傾き 180度) のまわりで勾配が消えてしまい、最初の数度を
            # 揺すり出す動機が出ない。復帰の全行程に一定の勾配がある
            # 唯一の項なので、ここだけは角度で測る。
            "upright": 1.0 - jp.arccos(jp.clip(up, -1.0, 1.0)) / jp.pi,
            "height": gate * jp.exp(
                -jp.square(h_rew - h_tgt) / cc.height_sigma),
            "height_low": up_pos * jp.clip(h_tgt - h_rew, 0.0, None),
            "feet_down": gate * jp.mean(contact_f),
            "recovered": ok,
            "stand_pose": (up > cc.upright_cos) * jp.sum(
                jp.square(q - self._default_pose)),
            # ゲートの選択: rec2 は up_pos (伏せでも 0.77 倍効いて立ち上がりを
            # 抑制)、rec3 は ok (復帰後のみ) で試したが、どちらも 8秒間立ち
            # 続けられたのは 10.7% で差が出なかった。立ち姿勢の維持には
            # 効いたほうがよいので、両者の中間 (正立に十分近いときだけ) にする。
            "settle": (up > 0.7) * (jp.sum(jp.square(v))
                                    + 0.1 * jp.sum(jp.square(w))),
            "trunk_contact": trunk,
            "rock": jp.clip(up - info["best_up"], 0.0, None),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "torque_limit": jp.sum(jp.square(over)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            "dof_pos_limits": out_of_limits,
            "linkage_infeasible": infeasibility(q),
        }
