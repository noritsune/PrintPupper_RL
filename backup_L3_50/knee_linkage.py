"""PrintPupper v0.2 の非平行リンク膝機構。
pp_champ_joints_hardware/src/ComputeUnparallelLinkKnee.cpp の Python 移植。

膝サーボは股関節付近に固定され、4節リンクで下腿を駆動する。
よって「関節角 L2」と「サーボ角 L3」は非線形な対応になる。
RL 側は関節角で学習し、実機ノードがこの写像でサーボ角に変換する。
"""
import numpy as np

L1 = 110.0        # 大腿長 [mm]  (URDF: upper->lower = 0.11 m)
L2_LEN = 120.0    # 下腿長 [mm]  (URDF: lower->foot = 0.12 m)
L2_MID = 60.0     # 下腿上の連結点
L3_LEN = 50.0     # サーボホーン
L4_LEN = 90.0     # プッシュロッド
OFF_X, OFF_Y = 5.0, -25.0   # 大腿原点から見たサーボ軸位置

EPS = 1e-6


def _norm(r):
    return (r + np.pi) % (2 * np.pi) - np.pi


def compute(l1_rad, l2_rad):
    """関節角 (大腿 l1, 膝相対 l2) -> 膝サーボ角 L3。解なしは None。"""
    l1a = l1_rad - np.pi / 2
    l2a = l1a + l2_rad
    gx, gy = L1 * np.cos(l1a), L1 * np.sin(l1a)
    yx, yy = gx + L2_MID * np.cos(l2a), gy + L2_MID * np.sin(l2a)
    dx, dy = yx - OFF_X, yy - OFF_Y
    d = np.hypot(dx, dy)
    if d < EPS or d > L3_LEN + L4_LEN + EPS or d < abs(L3_LEN - L4_LEN) - EPS:
        return None
    a = (L3_LEN**2 - L4_LEN**2 + d * d) / (2 * d)
    h2 = L3_LEN**2 - a * a
    if h2 < -EPS:
        return None
    h = np.sqrt(max(h2, 0.0))
    ux, uy = dx / d, dy / d
    px, py = OFF_X + a * ux, OFF_Y + a * uy
    cands = [(px - uy * h, py + ux * h), (px + uy * h, py - ux * h)]
    rads = [_norm(np.arctan2(by - OFF_Y, bx - OFF_X) + np.pi / 2) for bx, by in cands]
    l2dir = _norm(l2a + np.pi / 2)
    return min(rads, key=lambda r: abs(_norm(r - l2dir)))


# --- サーボ諸元 (PpChampJointsHardware.cpp より) ---
PWM_US_MIN, PWM_US_NEUTRAL, PWM_US_MAX = 500, 1500, 2500
US_PER_RAD = (PWM_US_MAX - PWM_US_MIN) / np.pi      # 636.62 us/rad
SERVO_HALF_RANGE = (PWM_US_MAX - PWM_US_NEUTRAL) / US_PER_RAD   # = pi/2
NEUTRAL = {"hip": 0.0, "upper": np.pi / 4, "lower": -np.pi / 4}


def servo_cmd_ok(joint_rad, which):
    """サーボ指令が PWM 飽和せずに出せるか。"""
    return abs(joint_rad - NEUTRAL[which]) <= SERVO_HALF_RANGE + 1e-9
