"""実機デプロイ用。学習時の観測組立とサーボ変換を NumPy で忠実に再現する。

ここが学習時と1ビットでもずれると実機で動かない。env/joystick.py の
_get_obs と同じ順序・同じスケールであることを scripts/check_deploy.py で
数値的に突き合わせている。

実機で必要なもの:
  - IMU (6軸: ジャイロ + 加速度計)。これ以外のセンサは使わない。
  - 姿勢推定 (相補フィルタ / Madgwick / MPU6050 の DMP)。重力方向が取れればよい。
  - 内部クロック (歩容位相用)

ROS2 ノードの1ステップ:
  IMU購読 -> ObsBuilder.step() -> onnxruntime 推論 -> joint_targets()
  -> ServoMap.joint_to_servo() -> PWM 出力
"""
import numpy as np

# --- 学習時と一致させる定数 (env/constants.py, env/joystick.py 由来) ---
DEFAULT_POSE = np.array([0.0, np.pi / 4, -np.pi / 2] * 4, dtype=np.float32)
ACTION_SCALE = 0.25
ACCEL_OBS_SCALE = 0.1
CTRL_DT = 0.02          # 50 Hz
OBS_DIM = 38            # 観測履歴なしのときの次元 (obs_dim() を使うこと)

# --- 観測履歴スタック (env/joystick.py の obs_history の写し) ---
# 1 フレーム = [gyro 3, accel*ACCEL_OBS_SCALE 3, gravity 3, action 12] = 21。
# ObsBuilder(history_len=L, history_stride=S) で有効になり、観測は
# 38 + L*21 次元になる。**学習時と同じ L / S を渡すこと。**
# 値がずれても次元が合ってしまう組み合わせがあるので (例 L=10,S=4 と L=10,S=2)、
# scripts/export_npz.py が .npz に書き込んだ値を使うのが安全。
HIST_FRAME_DIM = 21


def obs_dim(history_len: int = 0) -> int:
    return OBS_DIM + int(history_len) * HIST_FRAME_DIM

# 転倒復帰 (env/recover.py) の残差スケール。
# 歩行の 0.25 では、仰向けの機体から足先が床に届かない (足先の到達は機体
# フレーム z=+49mm 止まりで、床は +57mm)。復帰方策だけはここを使うこと:
#   目標角 = DEFAULT_POSE + action * RECOVER_ACTION_SCALE
# 基準姿勢は歩行と同じ 4脚接地姿勢なので、復帰後に action=0 を出せば
# そのまま歩行方策へ引き渡せる。
RECOVER_ACTION_SCALE = 1.6

# --- 段差歩行 + 転倒復帰を 1 つの方策でやる版 (env/unified.py) ---
# 可動域を胴体の傾きの関数にする。値は env/constants.py の写し
# (実機側を numpy だけで動かすため。変えたら両方を直すこと。
#  scripts/check_deploy.py が一致を検査する)。
#   up = -gravity[2]   (+1 正立 / 0 横倒し / -1 仰向け)
#   a  = smoothstep(up; 0.50, 0.90)
#   残差スケール = 1.6 + (0.25 - 1.6) * a
# 歩行中 (傾き 26度以内) は 0.25 ちょうど、転倒中は 1.6 ちょうどになる。
# IMU の姿勢推定誤差 (数度) では帯をまたがない。
GATE_UP_LO = 0.50
GATE_UP_HI = 0.90
# a がこれ以下なら「転んでいる」とみなし、速度コマンドを無視して歩容位相を
# 止める (学習時の観測と揃えるため。env/unified.py の _get_obs と同じ規則)。
GATE_OBS_SWITCH = 0.5


def upright_gate(gravity) -> float:
    """重力方向 -> 1.0 正立 / 0.0 転倒 の滑らかなゲート (smoothstep)。"""
    g = np.asarray(gravity, np.float32)
    up = float(-g[2] / (np.linalg.norm(g) + 1e-6))
    t = min(max((up - GATE_UP_LO) / (GATE_UP_HI - GATE_UP_LO), 0.0), 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def unified_action_scale(gravity) -> float:
    """その姿勢で使う残差スケール [rad]。"""
    a = upright_gate(gravity)
    return RECOVER_ACTION_SCALE + (ACTION_SCALE - RECOVER_ACTION_SCALE) * a


# 「伏せ」姿勢 (前脚を前に投げ出した低い姿勢) の基準姿勢からの差分 [rad]。
# env/recover.py の recover.prone_delta の写し。学習時の初期姿勢のひとつで、
# scripts/view.py --recover --prone がこの姿勢から始める。
RECOVER_PRONE_DELTA = np.array([0.0, -0.7, -0.6], dtype=np.float32)

# 対角2脚での静止保持 (env/balance.py) の基準姿勢。
# この課題だけ「基準姿勢」が4脚接地姿勢ではなくトロット中間姿勢になる:
#   目標角 = balance_pose(pair) + action * ACTION_SCALE
# 値は env/constants.BALANCE_LIFT の写し (実機側を numpy だけで動かすため)。
# scripts/view.py が起動時に一致を検査する。
BALANCE_LIFT = np.array([0.0, 0.40, -0.45], dtype=np.float32)
BALANCE_PAIRS = {"a": np.array([1.0, 0.0, 0.0, 1.0], np.float32),   # lf+rh 接地
                 "b": np.array([0.0, 1.0, 1.0, 0.0], np.float32)}   # rf+lh 接地


def balance_pose(pair="a"):
    """対角2脚保持の基準姿勢 (12関節 [rad])。pair='a' は lf+rh で立つ。"""
    swing = 1.0 - BALANCE_PAIRS[pair]
    return (DEFAULT_POSE.reshape(4, 3) + BALANCE_LIFT * swing[:, None]).ravel()


def balance_command(pair="a"):
    """観測の速度コマンド枠に入れる値。balance では対角ペアの指定に転用する。"""
    return np.array([1.0 if pair == "a" else -1.0, 0.0, 0.0], np.float32)

JOINT_ORDER = [f"{l}_{j}_joint" for l in ("lf", "rf", "lh", "rh")
               for j in ("hip", "upper_leg", "lower_leg")]


class ObsBuilder:
    """観測ベクトルを組み立てる。実機と sim の両方で同じコードを使う。"""

    def __init__(self, gait_freq: float = 1.5, nominal_pose=None,
                 action_scale: float = ACTION_SCALE, unified: bool = False,
                 history_len: int = 0, history_stride: int = 4):
        # nominal_pose: 行動残差の基準になる姿勢。歩行は4脚接地の基準姿勢だが、
        # 対角2脚保持 (env/balance.py) は balance_pose(pair) を渡す。
        # unified=True: 段差歩行 + 転倒復帰を 1 つの方策でやる版
        # (env/unified.py)。action_scale・速度コマンド・歩容位相を、直前に
        # 読んだ IMU の傾きで切り替える。IMU 以外のセンサは要らない。
        self.default_pose = (DEFAULT_POSE if nominal_pose is None
                             else np.asarray(nominal_pose, np.float32))
        self.gait_freq = gait_freq
        self.action_scale = float(action_scale)
        self.unified = bool(unified)
        # 観測履歴スタック。学習時の cfg.obs_history と同じ値を渡すこと。
        self.history_len = int(history_len)
        self.history_stride = max(1, int(history_stride))
        self._hist_frames = self.history_len * self.history_stride
        self._hist_idx = (np.arange(1, self.history_len + 1)
                          * self.history_stride - 1)
        self.obs_dim = obs_dim(self.history_len)
        self.reset()

    def reset(self):
        self.last_act = np.zeros(12, np.float32)
        self.last_last_act = np.zeros(12, np.float32)
        self.phase = 0.0
        # 履歴のリングバッファ。hist[0] が 1 ステップ前。**電源投入直後は
        # 1 回目の読み値でバッファ全体を埋める** (学習側 env/joystick.py の
        # _obs_reset と同じ)。ゼロ埋めのまま始めると、重力 z の std が 0.0188
        # しかないせいで正規化後に +52.7σ の入力になり、学習時に見たことの
        # ない領域を最初の 0.8 秒だけ通ることになる。
        self._hist = None          # 1 回目の step() で「いまの読み値」で埋める
        self._last_sensors = np.zeros(9, np.float32)
        # unified: 直近の step() が見た姿勢から決めた値。commit() が使う。
        self._scale = self.action_scale if not getattr(self, "unified", False) \
            else RECOVER_ACTION_SCALE
        self._upright = not getattr(self, "unified", False)

    def step(self, gyro, accel, gravity, command):
        """1制御ステップぶんの観測を返す。

        gyro     (3) IMU 角速度 [rad/s] 機体座標
        accel    (3) IMU 加速度計の生値 (比力) [m/s^2] 機体座標
        gravity  (3) 姿勢推定が出す重力方向の単位ベクトル (機体座標, 下向きが -z)
        command  (3) [vx, vy, wz]
        """
        g = np.asarray(gravity, np.float32)
        g = g / (np.linalg.norm(g) + 1e-6)
        if self.unified:
            # 転んでいる間は「コマンド 0 / 位相 0」に固定する。学習時の
            # 観測 (env/unified.py の _get_obs) と同じ規則。
            self._scale = unified_action_scale(g)
            self._upright = upright_gate(g) > GATE_OBS_SWITCH
            if not self._upright:
                command = np.zeros(3, np.float32)
                self.phase = 0.0
        ph = 2 * np.pi * self.phase
        self._last_sensors = np.concatenate([
            np.asarray(gyro, np.float32),
            np.asarray(accel, np.float32) * ACCEL_OBS_SCALE,
            g,
        ]).astype(np.float32)
        if self.history_len > 0 and self._hist is None:
            # 電源投入直後。「この姿勢で 0.8 秒静止していた」という履歴にする。
            frame = np.concatenate([self._last_sensors, self.last_act])
            self._hist = np.tile(frame, (self._hist_frames, 1)).astype(np.float32)
        obs = [
            self._last_sensors,                                  # 9
            np.asarray(command, np.float32),                     # 3
            self.last_act,                                       # 12
            self.last_last_act,                                  # 12
            np.array([np.cos(ph), np.sin(ph)], np.float32),      # 2
        ]
        if self.history_len > 0:
            obs.append(self._hist[self._hist_idx].ravel())
        obs = np.concatenate(obs).astype(np.float32)
        # 履歴は観測を作った **後** に詰める (obs が自分自身のフレームを
        # 見ないようにする)。詰める行動は「いま読んだセンサ値と対になる行動」=
        # **self.last_act (1 つ前に出した指令)**。いまから出す行動ではない:
        # 学習側 (env/joystick.py の step) はフレームを
        #   (a_t を適用した後のセンサ, a_t)
        # の組で詰めており、実機でその a_t にあたるのは last_act だから。
        # ここを commit(action) 側で詰めると 1 ステップぶん先の行動と組になり、
        # 履歴だけが学習時と食い違う (観測の先頭 38 次元は一致したままなので
        # 気付きにくい。scripts/check_deploy.py の obs_history 行で検出できる)。
        if self.history_len > 0:
            frame = np.concatenate([self._last_sensors, self.last_act])
            self._hist = np.roll(self._hist, 1, axis=0)
            self._hist[0] = frame
        return obs

    def commit(self, action):
        """推論結果を記録し、位相を進める。step() の後に必ず呼ぶ。

        学習側は「行動を出した後」に観測を作る (last_act = 今出した a_t)。
        よって実機ループも   obs = step(...) -> action = policy(obs)
        -> joint = commit(action)  の順で、次の step() が a_t を見る形になる。
        """
        action = np.asarray(action, np.float32)
        self.last_last_act = self.last_act
        self.last_act = action
        # unified: 転んでいる間は位相クロックを進めない (学習時と同じ)。
        if not (self.unified and not self._upright):
            self.phase = (self.phase + self.gait_freq * CTRL_DT) % 1.0
        return self.joint_targets(action)

    def joint_targets(self, action):
        """方策出力 -> 関節角指令 [rad] (基準姿勢からの残差)。

        unified では残差スケールが姿勢で変わる。使うのは **直前の step() が
        見た IMU の値**から決めたもの: その観測から出た行動に、その観測と
        同じ姿勢のスケールを掛ける (学習側の env/unified.py と同じ組み合わせ)。
        """
        scale = self._scale if self.unified else self.action_scale
        return self.default_pose + np.asarray(action, np.float32) * scale


# --------------------------------------------------------------------------
# 4節リンク膝 + PWM サーボ変換
# pp_champ_joints_hardware/src/{ComputeUnparallelLinkKnee,PpChampJointsHardware}.cpp
# --------------------------------------------------------------------------
_L1, _L2, _L2MID, _L3, _L4 = 110.0, 120.0, 60.0, 50.0, 90.0
_OFFX, _OFFY = 5.0, -25.0

PWM_HZ = 100
PWM_US_MIN, PWM_US_NEUTRAL, PWM_US_MAX = 500, 1500, 2500
US_PER_RAD = (PWM_US_MAX - PWM_US_MIN) / np.pi

# 脚の並び: C++ 側は FL, FR, BL, BR。学習側は lf, rf, lh, rh。同じ順序。
SERVO_NEUTRAL = np.array([[0.0,  np.pi/4, -np.pi/4]] * 4)
SERVO_INVERT = np.array([
    [ 1.0,  1.0,  1.0],   # FL
    [ 1.0, -1.0, -1.0],   # FR
    [-1.0,  1.0,  1.0],   # BL
    [-1.0, -1.0, -1.0],   # BR
])


def _norm(r):
    return (r + np.pi) % (2 * np.pi) - np.pi


def knee_servo_angle(l1, l2):
    """関節角 (大腿, 膝相対) -> 膝サーボ角。解なしは None。"""
    l1a = l1 - np.pi / 2
    l2a = l1a + l2
    gx, gy = _L1 * np.cos(l1a), _L1 * np.sin(l1a)
    yx, yy = gx + _L2MID * np.cos(l2a), gy + _L2MID * np.sin(l2a)
    dx, dy = yx - _OFFX, yy - _OFFY
    d = np.hypot(dx, dy)
    if d < 1e-6 or d > _L3 + _L4 or d < abs(_L3 - _L4):
        return None
    a = (_L3**2 - _L4**2 + d * d) / (2 * d)
    h = np.sqrt(max(_L3**2 - a * a, 0.0))
    ux, uy = dx / d, dy / d
    px, py = _OFFX + a * ux, _OFFY + a * uy
    r1 = _norm(np.arctan2(py + ux * h - _OFFY, px - uy * h - _OFFX) + np.pi / 2)
    r2 = _norm(np.arctan2(py - ux * h - _OFFY, px + uy * h - _OFFX) + np.pi / 2)
    l2dir = _norm(l2a + np.pi / 2)
    return r1 if abs(_norm(r1 - l2dir)) <= abs(_norm(r2 - l2dir)) else r2


def joint_to_servo(joint_rad, calib_deg=None):
    """12関節角 [rad] -> 12サーボ角 [rad]。膝は4節リンクを通す。

    calib_deg: ServoCalibration.conf の中立オフセット (4x3, deg)。
    """
    q = np.asarray(joint_rad, np.float32).reshape(4, 3)
    out = np.empty((4, 3), np.float32)
    prev_knee = None
    for i in range(4):
        out[i, 0] = q[i, 0]
        out[i, 1] = q[i, 1]
        l3 = knee_servo_angle(q[i, 1], q[i, 2])
        # リンク解が無い姿勢は実機では出せない。直前値を保持する
        # (C++ 側も L3RadSafe=0 のまま据え置く実装になっている)
        out[i, 2] = l3 if l3 is not None else (
            prev_knee if prev_knee is not None else SERVO_NEUTRAL[i, 2])
        prev_knee = out[i, 2]
    neutral = SERVO_NEUTRAL.copy()
    if calib_deg is not None:
        neutral = neutral - np.radians(np.asarray(calib_deg)) * SERVO_INVERT
    return (out - neutral) * SERVO_INVERT


def servo_to_pwm_us(servo_rad):
    """サーボ角 -> PWM パルス幅 [us]。C++ posRadDutycycle と同じ。"""
    us = PWM_US_NEUTRAL + US_PER_RAD * np.asarray(servo_rad)
    return np.clip(us, PWM_US_MIN, PWM_US_MAX)


def pwm_saturated(servo_rad):
    """PWM 飽和している関節のマスク。実機で指令どおり動かない箇所を示す。"""
    us = PWM_US_NEUTRAL + US_PER_RAD * np.asarray(servo_rad)
    return (us < PWM_US_MIN) | (us > PWM_US_MAX)
