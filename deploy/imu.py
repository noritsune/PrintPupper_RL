"""MPU6050 / MPU9250 の読み出しと、重力方向の推定。

観測に要るのは3つ ([env/deploy.py] の ObsBuilder.step):
  gyro    (3) 角速度 [rad/s]  機体座標
  accel   (3) 加速度計の生値 (比力) [m/s^2] 機体座標
  gravity (3) 重力方向の単位ベクトル (機体座標)。正立で [0, 0, -1]

機体座標は MuJoCo の base_link と同じ **x 前方 / y 左 / z 上**。
IMU サイトは base_link 原点にある (model/pupper_unified.xml の <site name="imu">)
ので、チップの向きの違いは AXIS_MAP / AXIS_SIGN だけで吸収できる。

学習時に想定している誤差幅 (env/joystick.py の imu_config):
  ジャイロ白色 0.05 rad/s / 加速度白色 0.30 m/s^2 / 重力方向 白色 0.02
  **傾きの定常バイアス 0.03 rad (1.7度)**  <- 取り付け角ずれ込み。ここが一番効く。

統合方策は重力方向だけで可動域を 0.25 <-> 1.6 rad と切り替える
(env/deploy.py の upright_gate) ので、1.7度を超える定常ずれがあると
歩行中に復帰側の可動域へ落ちることがある。水平な床で実測して確認すること。
"""
import time

import numpy as np

# --- レジスタ (MPU6050 / MPU9250 共通部分) ---
ADDR = 0x68
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B      # accel 6 + temp 2 + gyro 6 = 14 バイト連続
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75
WHO_AM_I_OK = {0x68: "MPU6050", 0x70: "MPU6500", 0x71: "MPU9250", 0x73: "MPU9255"}

G = 9.80665

# フルスケール。復帰の蹴り出しと着地を切らないように広めに取る。
ACCEL_FS_G = 8               # +-8g   -> 4096 LSB/g
GYRO_FS_DPS = 1000           # +-1000 dps -> 32.8 LSB/dps (= 17.5 rad/s)
_ACCEL_CONF = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}
_GYRO_CONF = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}
_ACCEL_LSB = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
_GYRO_LSB = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}

# DLPF。44Hz (accel) / 42Hz (gyro)、群遅延 4.9 / 4.8 ms。
# 200 Hz で読むので折り返しは出ない。遅延は制御周期 20ms の 1/4 で、
# 学習側の action_delay (1-3 step = 20-60ms) の枠に収まる。
DLPF_CFG = 3

# --- 取り付け ---
# チップの軸 (x,y,z) を機体座標のどの軸に割り当てるか。
# AXIS_MAP[i] = 機体座標 i 番目に入れるチップの軸番号。AXIS_SIGN でさらに反転。
# 例: チップを 180度 回して載せたなら AXIS_SIGN = (-1, -1, 1)。
# **ここを間違えると静止指令でいきなり倒れる**。吊り試験で grav を目視すること。
AXIS_MAP = (0, 1, 2)
AXIS_SIGN = (1.0, 1.0, 1.0)


class Imu:
    """I2C から読んで、重力方向をジャイロ主体の相補フィルタで推定する。

    復帰の蹴り出し (0.5-1.0s) は加速度が大きく、加速度計を信じるフィルタは
    そこで姿勢を見失う。そこで
      - 基本はジャイロの積分で伝搬させ、
      - |accel| が 1g から離れているステップでは加速度の補正を切る
    という形にしてある。ACCEL_TRUST は 200Hz あたりの混合率。
    """

    ACCEL_TRUST = 0.02        # 時定数 約 0.25s @200Hz
    ACCEL_GATE = 2.0          # |accel| が 1g から何 m/s^2 離れたら補正を切るか

    def __init__(self, bus=1, addr=ADDR, axis_map=AXIS_MAP, axis_sign=AXIS_SIGN):
        from smbus2 import SMBus
        self._bus = SMBus(bus)
        self._addr = addr
        self._map = np.asarray(axis_map, int)
        self._sign = np.asarray(axis_sign, np.float32)

        who = self._bus.read_byte_data(addr, REG_WHO_AM_I)
        self.chip = WHO_AM_I_OK.get(who)
        if self.chip is None:
            raise RuntimeError(
                f"WHO_AM_I = 0x{who:02x} は既知の MPU ではない "
                f"(期待: {', '.join(f'0x{k:02x}' for k in WHO_AM_I_OK)})")

        self._bus.write_byte_data(addr, REG_PWR_MGMT_1, 0x80)   # reset
        time.sleep(0.1)
        self._bus.write_byte_data(addr, REG_PWR_MGMT_1, 0x01)   # PLL (gyro X)
        time.sleep(0.01)
        self._bus.write_byte_data(addr, REG_CONFIG, DLPF_CFG)
        self._bus.write_byte_data(addr, REG_SMPLRT_DIV, 0x00)   # 内部 1kHz
        self._bus.write_byte_data(addr, REG_GYRO_CONFIG, _GYRO_CONF[GYRO_FS_DPS])
        self._bus.write_byte_data(addr, REG_ACCEL_CONFIG, _ACCEL_CONF[ACCEL_FS_G])
        time.sleep(0.05)

        self._a_scale = G / _ACCEL_LSB[ACCEL_FS_G]
        self._g_scale = np.radians(1.0) / _GYRO_LSB[GYRO_FS_DPS]
        self.gyro_bias = np.zeros(3, np.float32)

        self.gyro = np.zeros(3, np.float32)
        self.accel = np.array([0.0, 0.0, G], np.float32)
        self.gravity = np.array([0.0, 0.0, -1.0], np.float32)
        self.accel_rejected = 0     # 加速度補正を切った回数 (診断用)

    # -- 生の読み出し ------------------------------------------------------
    def _raw(self):
        d = self._bus.read_i2c_block_data(self._addr, REG_ACCEL_XOUT_H, 14)
        v = np.frombuffer(bytes(d), dtype=">i2")        # ビッグエンディアン 7 語
        ax, ay, az, _temp, gx, gy, gz = v
        accel = np.array([ax, ay, az], np.float32) * self._a_scale
        gyro = np.array([gx, gy, gz], np.float32) * self._g_scale
        return self._remap(gyro), self._remap(accel)

    def _remap(self, v):
        return (v[self._map] * self._sign).astype(np.float32)

    # -- 校正 --------------------------------------------------------------
    def calibrate_gyro(self, secs=2.0, rate=200.0):
        """静止させた状態でジャイロのバイアスを測って差し引く。

        機体は動かさないこと。あわせて、そのときの加速度から重力方向の初期値を
        取り、|accel| が 1g からどれだけずれているかを返す (加速度計の校正の目安)。
        """
        n = int(secs * rate)
        gs, as_ = np.zeros(3), np.zeros(3)
        for _ in range(n):
            g, a = self._raw()
            gs += g
            as_ += a
            time.sleep(1.0 / rate)
        self.gyro_bias = (gs / n).astype(np.float32)
        a_mean = (as_ / n).astype(np.float32)
        self.gravity = (-a_mean / np.linalg.norm(a_mean)).astype(np.float32)
        return {
            "gyro_bias": self.gyro_bias.copy(),
            "accel_norm": float(np.linalg.norm(a_mean)),
            "gravity0": self.gravity.copy(),
            # 正立で置いたなら、この値が取り付け角ずれ + 推定誤差そのもの。
            # 学習の想定は 0.03 rad (1.7度) 以内。
            "tilt_deg": float(np.degrees(np.arccos(
                np.clip(-self.gravity[2], -1.0, 1.0)))),
        }

    # -- 1 サンプル進める --------------------------------------------------
    def update(self, dt):
        """I2C を1回読んで相補フィルタを dt だけ進める。"""
        gyro, accel = self._raw()
        gyro = gyro - self.gyro_bias

        # 機体が w で回ると、固定方向ベクトルの機体座標表現は -w x g で動く
        g = self.gravity - np.cross(gyro, self.gravity) * dt

        # 加速度が 1g から離れているステップは信じない (復帰の蹴り出し対策)
        norm = float(np.linalg.norm(accel))
        if abs(norm - G) < self.ACCEL_GATE and norm > 1e-3:
            g = (1.0 - self.ACCEL_TRUST) * g + self.ACCEL_TRUST * (-accel / norm)
        else:
            self.accel_rejected += 1

        self.gravity = (g / (np.linalg.norm(g) + 1e-9)).astype(np.float32)
        self.gyro = gyro
        self.accel = accel
        return self.gyro, self.accel, self.gravity

    def close(self):
        self._bus.close()
