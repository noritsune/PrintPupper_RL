"""4節リンク膝機構の JAX 実装 (scripts/knee_linkage.py と数値一致)。

膝サーボは股元に固定され、リンクを介して下腿を駆動する。したがって
「MJCF の関節角の矩形リミット」は実機の可動域と一致しない。学習中の姿勢が
実機で出せるかをここで評価し、報酬ペナルティとして与える。
"""
import jax.numpy as jp

L1, L2_LEN, L2_MID, L3_LEN, L4_LEN = 110.0, 120.0, 60.0, 50.0, 90.0
OFF_X, OFF_Y = 5.0, -25.0
KNEE_SERVO_NEUTRAL = -jp.pi / 4
SERVO_HALF_RANGE = jp.pi / 2          # PWM 500-2500us / 636.6us-per-rad


def _norm(r):
    return (r + jp.pi) % (2 * jp.pi) - jp.pi


def knee_servo_angle(l1, l2):
    """関節角 -> 膝サーボ角 L3 と、リンク解が存在するかのフラグ。"""
    l1a = l1 - jp.pi / 2
    l2a = l1a + l2
    gx, gy = L1 * jp.cos(l1a), L1 * jp.sin(l1a)
    yx, yy = gx + L2_MID * jp.cos(l2a), gy + L2_MID * jp.sin(l2a)
    dx, dy = yx - OFF_X, yy - OFF_Y
    d = jp.sqrt(dx * dx + dy * dy)
    solvable = (d > 1e-6) & (d <= L3_LEN + L4_LEN) & (d >= abs(L3_LEN - L4_LEN))
    d_safe = jp.where(solvable, d, 1.0)
    a = (L3_LEN**2 - L4_LEN**2 + d_safe**2) / (2 * d_safe)
    h = jp.sqrt(jp.clip(L3_LEN**2 - a * a, 0.0, None))
    ux, uy = dx / d_safe, dy / d_safe
    px, py = OFF_X + a * ux, OFF_Y + a * uy
    r1 = _norm(jp.arctan2(py + ux * h - OFF_Y, px - uy * h - OFF_X) + jp.pi / 2)
    r2 = _norm(jp.arctan2(py - ux * h - OFF_Y, px + uy * h - OFF_X) + jp.pi / 2)
    l2dir = _norm(l2a + jp.pi / 2)
    pick1 = jp.abs(_norm(r1 - l2dir)) <= jp.abs(_norm(r2 - l2dir))
    return jp.where(pick1, r1, r2), solvable


def infeasibility(qpos12):
    """12 関節角ベクトル -> 実機で出せない度合い [rad 相当]。0 なら完全に実現可能。

    (a) リンク解が存在しない姿勢、(b) 膝サーボ指令が PWM 飽和する姿勢 を罰する。
    """
    q = qpos12.reshape(4, 3)
    l3, solvable = knee_servo_angle(q[:, 1], q[:, 2])
    over = jp.clip(jp.abs(l3 - KNEE_SERVO_NEUTRAL) - SERVO_HALF_RANGE, 0.0, None)
    # 解なしは最大逸脱として扱う
    return jp.sum(jp.where(solvable, over, SERVO_HALF_RANGE))
