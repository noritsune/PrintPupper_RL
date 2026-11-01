"""PrintPupper の MJCF を生成する。

なぜ URDF 直接変換ではなくスクリプト生成か:
  quadruped.urdf は CHAMP 用の「運動学のみ」の記述で、<inertial> も <collision> も
  一切含まれない。MuJoCo は可動ボディに質量ゼロを許さずコンパイル時点で落ちる。
  よって運動学(検証済み)は URDF から取り、慣性は実メッシュ形状から算出し、
  接触形状・アクチュエータはここで新規に与える。

質量配分の根拠:
  - hip_link は 70x71x55mm の筐体で、脚サーボと膝サーボの2個を内蔵する
    (膝が4節リンク駆動なのは膝サーボが股元にあるため)。標準サイズサーボ
    (MG996R 級 55g) 2個 + 印刷部品 で 135g と見積もる。
  - 静止時の必要膝トルクは 0.28 Nm。MG90S(0.2Nm)では保持できず、
    MG996R級(0.9-1.1Nm)が必要 -- サーボ級の推定はこの静力学とも整合する。
"""
import numpy as np, trimesh, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env import constants as consts    # 地形寸法はここを唯一の出典にする

MESH = os.path.join(os.path.dirname(__file__), "..", "model", "meshes")

# --- 質量配分 [kg] (実機実測で要更新) ---
# DS3235 は 60g / 40x20x40.5mm。12個で 720g あり、質量の半分近くがサーボ。
#   base_link : 印刷筐体 ~200g + RPi 45g + 2S電池 ~120g + 配線/IMU 40g
#               + 股関節サーボ 4個 240g = 645g
#   hip_link  : 脚サーボ + 膝サーボ の2個 120g + 印刷ブラケット 30g
#               (膝が4節リンク駆動なのは膝サーボがここに載るため)
M_BODY, M_HIP, M_UPPER, M_LOWER = 0.645, 0.150, 0.032, 0.026

# --- 運動学 (quadruped.urdf より) ---
HIP_X, HIP_Y = 0.10, 0.045      # base -> hip
ABD_Y        = 0.045            # hip -> upper leg (外側オフセット)
L_UPPER      = 0.11             # upper -> lower
L_LOWER      = 0.12             # lower -> foot
FOOT_R       = 0.012

# --- 基準姿勢 (gait.yaml の nominal_height / com_x_translation と一致を確認済み) ---
Q_HIP, Q_UPPER, Q_LOWER = 0.0, np.pi / 4, -np.pi / 2
STAND_H = 0.1626

# --- 関節リミット ---
# hip/upper: サーボ PWM 500-2500us (= neutral +-90deg) から直接決まる
# lower: 4節リンク経由なので upper と連成する。ここには包絡線(必ず不可能な外側)を
#        入れ、連成拘束そのものは RL 側のペナルティで扱う。CONSERVATIVE は
#        upper 0-90deg 全域で無条件に安全な箱(学習初期用フォールバック)。
R_HIP   = (-np.pi / 2, np.pi / 2)
R_UPPER = (-np.pi / 4, 3 * np.pi / 4)
R_LOWER = (np.radians(-170.0), np.radians(-34.0))
R_LOWER_CONSERVATIVE = (np.radians(-122.0), np.radians(-71.0))

# --- アクチュエータ: DSSERVO DS3235 ---
# 諸元 (データシート):
#   ストールトルク  30 kg-cm @4.8-6V  /  35 kg-cm @6.8V  = 2.94 / 3.43 Nm
#   無負荷速度      0.17 s/60deg @4.8V / 0.13 s/60deg @6.8V = 6.16 / 8.05 rad/s
#   質量 60g, 寸法 40x20x40.5mm, デッドバンド 3us
#   PWM 500-2500us で 180deg  <- ハードウェアノードの PWM 設定と完全に一致する
#
# 6V 運用を公称とする (2S LiPo + BEC を想定)。
FORCERANGE = 3.0          # ストールトルク [Nm]
NOLOAD_SPEED = 7.0        # 無負荷角速度 [rad/s] (0.15 s/60deg 相当)

# damping はトルク-速度特性の1次近似 (背起電力). tau = tau_stall*(1 - w/w0) を
# 「一定トルク源 + 粘性」に分解すると粘性係数は tau_stall/w0 になる。
# これを入れないとサーボが無限に速く振れてしまい、実機で追従できない歩容を学習する。
DAMPING = FORCERANGE / NOLOAD_SPEED       # = 0.43 Nm*s/rad

# armature = ロータ慣性 x 減速比^2。DS3235 のロータを 8g/phi12mm、減速比 ~280 と
# 見て 1.4e-7 * 280^2 ~ 0.011。リンク慣性(3e-5)の数百倍で、ギヤードサーボでは
# これが支配的になるのが正しい。実機入手後に同定し直す対象。
ARMATURE = 0.011
FRICLOSS = 0.010          # ギヤのクーロン摩擦 (MG996R 級より大きいギヤ列)

# kp: デッドバンド 3us から出る剛性 (~640 Nm/rad) は数値的に扱えないので、
# 閉ループ帯域が実機相当 (~10Hz) になる値を採る。w_n = sqrt(kp/armature)。
KP = 50.0                 # -> w_n = sqrt(50/0.011) = 67 rad/s = 10.7 Hz
# kv: DS3235 はデジタルサーボで内部に D 項を持つ。臨界減衰から背起電力ぶんを
# 差し引いたぶんをアクチュエータ側の kv として与える (二重計上を避ける)。
KV = max(0.0, 2.0 * (KP * ARMATURE) ** 0.5 - DAMPING)

LEGS = [  # name, sx(前後), sy(左右), hip mesh, upper mesh
    ("lf",  1,  1, "pp_vm_hip_FL.stl", "pp_vm_upperleg_L.stl"),
    ("rf",  1, -1, "pp_vm_hip_FR.stl", "pp_vm_upperleg_R.stl"),
    ("lh", -1,  1, "pp_vm_hip_BL.stl", "pp_vm_upperleg_L.stl"),
    ("rh", -1, -1, "pp_vm_hip_BR.stl", "pp_vm_upperleg_R.stl"),
]
# 後脚の hip メッシュは BL/BR、大腿メッシュは左右別 (L/R)


def inertial(mesh_file, mass, mirror_y=False):
    """メッシュ形状から質量 mass の慣性を算出し MJCF の <inertial> 文字列を返す。"""
    m = trimesh.load(os.path.join(MESH, mesh_file))
    m.apply_scale(0.001)                       # mm -> m
    if mirror_y:
        m.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
    m.density = mass / m.volume                # 一様密度で目標質量に合わせる
    c = m.center_mass
    I = m.moment_inertia                       # 重心まわり
    # MuJoCo fullinertia の順: ixx iyy izz ixy ixz iyz
    fi = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    return (f'<inertial pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{mass:.4f}" '
            f'fullinertia="{" ".join(f"{v:.3e}" for v in fi)}"/>')


def build(conservative_knee=False, terrain=False, legs_collide=False):
    """terrain=True で床を plane -> hfield に差し替えた段差版を出す。

    legs_collide=True で大腿・下腿にも衝突形状 (カプセル) を付けた転倒復帰版を
    出す (env/recover.py)。歩行版は足球と胴体箱しか衝突しないので、脚は床を
    すり抜ける。歩行中は足先しか床に触れないので問題にならないが、仰向けから
    起き上がる動作では脚を床に着いて押すため、すり抜けると実機で成立しない
    動きを覚える。カプセルは足球より 17mm 短くしてあり、4脚接地の姿勢では
    床に触れない (= 歩行版と同じ接触になる)。

    geom 名 "floor" は変えないこと。接触センサ (geom2="floor") がそれで
    引いており、名前を変えると足の接地判定が全滅する。hfield の中身
    (高さデータ) は XML には書かず、env/terrain.py が実行時に流し込む。
    こうすると難易度を変えるのに XML を作り直さなくて済む。
    """
    rl = R_LOWER_CONSERVATIVE if conservative_knee else R_LOWER
    x = []
    A = x.append
    A('<mujoco model="printpupper">')
    A('  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>')
    A('  <option timestep="0.004" iterations="4" ls_iterations="8" integrator="implicitfast">')
    A('    <flag eulerdamp="disable"/>')
    A('  </option>')
    # hfield は足球1個あたり最大4接触を出す (実測)。平地の plane より多い。
    # 両立版 (env/unified.py: 段差地形 + 脚の衝突形状) は床に当たりうる geom が
    # 足球4 + 脚カプセル8 + 胴体箱 = 13 まで増えるので、geom_pairs をそれ以上に
    # 取る。下回ると接触が落ちて機体が床を抜ける。
    if terrain and legs_collide:
        # 床に当たりうる geom は 足球4 + 脚カプセル8 + 胴体箱1 + 胴体の角球8 = 21。
        ncp, ngp = 24, 24
    else:
        ncp, ngp = (8, 8) if terrain else (12, 12) if legs_collide else (5, 4)
    A(f'  <custom><numeric data="{ncp}" name="max_contact_points"/>'
      f'<numeric data="{ngp}" name="max_geom_pairs"/></custom>')
    A('')
    A('  <default>')
    A('    <default class="pupper">')
    A(f'      <joint type="hinge" armature="{ARMATURE}" damping="{DAMPING}" '
      f'frictionloss="{FRICLOSS}" limited="true"/>')
    A(f'      <position kp="{KP}" kv="{KV:.4f}" '
      f'forcerange="-{FORCERANGE} {FORCERANGE}"/>')
    A('      <default class="visual">')
    A('        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" '
      'material="part"/>')
    A('      </default>')
    A('      <default class="collision">')
    # contype=0/conaffinity=1 で「床とだけ」当たる。自己干渉は報酬側で扱い接触計算を節約
    A('        <geom contype="0" conaffinity="1" group="3" condim="3" '
      'friction="0.8 0.02 0.01" solref="0.015 1" solimp="0.9 0.95 0.001"/>')
    A('      </default>')
    A('    </default>')
    A('  </default>')
    A('')
    A('  <visual>')
    A('    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-20"/>')
    A('    <quality shadowsize="4096"/>')
    A('  </visual>')
    A('')
    A('  <asset>')
    A('    <material name="part" rgba="0.75 0.75 0.78 1"/>')
    A('    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" '
      'reflectance="0.1"/>')
    A('    <texture name="grid" type="2d" builtin="checker" width="512" height="512" '
      'rgb1="0.2 0.25 0.3" rgb2="0.28 0.33 0.38"/>')
    A('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" '
      'rgb2="0.05 0.06 0.1" width="256" height="256"/>')
    if terrain:
        # 高さデータは空 (全ゼロ) で出す。env/terrain.py が mj_model.hfield_data に
        # 直接書き込む。size = (半辺x, 半辺y, データ1.0の高さ, 基礎の厚み)
        A(f'    <hfield name="terrain" nrow="{consts.TERRAIN_N}" '
          f'ncol="{consts.TERRAIN_N}" '
          f'size="{consts.TERRAIN_HALF} {consts.TERRAIN_HALF} '
          f'{consts.TERRAIN_ZMAX} 0.5"/>')
    for f in sorted({m for _, _, _, m, _ in LEGS} | {m for _, _, _, _, m in LEGS} |
                    {"pp_vm_body.stl", "pp_vm_lowerleg_LR.stl"}):
        A(f'    <mesh name="{f[:-4]}" file="{f}" scale="0.001 0.001 0.001"/>')
    A('  </asset>')
    A('')
    A('  <worldbody>')
    A('    <light name="spot" pos="0 0 2.5" dir="0 0 -1" directional="true"/>')
    if terrain:
        A('    <geom name="floor" type="hfield" hfield="terrain" material="grid" '
          'contype="1" conaffinity="0" condim="3" friction="0.8 0.02 0.01"/>')
    else:
        A('    <geom name="floor" type="plane" size="0 0 0.05" material="grid" '
          'contype="1" conaffinity="0" condim="3" friction="0.8 0.02 0.01"/>')
    A(f'    <body name="base_link" pos="0 0 {STAND_H + FOOT_R:.4f}" childclass="pupper">')
    A('      <freejoint name="root"/>')
    A('      <camera name="track" pos="0.6 -1.0 0.4" xyaxes="1 0.6 0 -0.2 0.35 0.9" '
      'mode="trackcom"/>')
    A('      ' + inertial("pp_vm_body.stl", M_BODY))
    A('      <geom class="visual" mesh="pp_vm_body"/>')
    # 胴体の接触箱 (転倒判定用)。メッシュ bounds x[-135,150] y[+-64.5] z[-15,57] mm
    A('      <geom name="trunk_col" class="collision" type="box" '
      'size="0.1425 0.0645 0.036" pos="0.0075 0 0.021"/>')
    if terrain and legs_collide:
        # **hfield の床では箱は 1 点でしか支えられない** (mujoco_warp の
        # MULTICCD は HFIELD-BOX を多点化できず「At most 1 contact」になる。
        # CPU の MuJoCo は三角形分割で 50 点出すので、ここは warp 固有の制約)。
        # 歩行では胴体が床に触れないので無害だが、仰向けの機体は胴体の上面
        # 全体で床に乗るため、1 点だと機体がその点を軸に転がるだけになり、
        # **脚で床を蹴って起き上がる動作が成立しない** (実測: 同じ方策
        # rec9 の復帰成功率が plane 床の 100% に対し hfield 床では 31%)。
        # 箱の8隅に、各面に内接する球を置いて多点支持を回復させる。
        # HFIELD-SPHERE は多点接触に対応している (足球が 1個あたり 4 点出す)。
        # 球は箱の面にちょうど内接するので箱より外へは出ず、段差の壁への
        # めり込みは今までどおり箱が止める。
        r = 0.012
        bs, bp = (0.1425, 0.0645, 0.036), (0.0075, 0.0, 0.021)
        k = 0
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    c = (bp[0] + sx * (bs[0] - r), bp[1] + sy * (bs[1] - r),
                         bp[2] + sz * (bs[2] - r))
                    A(f'      <geom name="trunk_c{k}" class="collision" '
                      f'type="sphere" size="{r}" '
                      f'pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"/>')
                    k += 1
    A('      <site name="imu" pos="0 0 0" size="0.008"/>')

    for name, sx, sy, hip_mesh, up_mesh in LEGS:
        S = "      "
        A('')
        A(f'{S}<body name="{name}_hip_link" pos="{HIP_X*sx:.3f} {HIP_Y*sy:.3f} 0">')
        A(f'{S}  <joint name="{name}_hip_joint" axis="1 0 0" '
          f'range="{R_HIP[0]:.4f} {R_HIP[1]:.4f}"/>')
        A(f'{S}  ' + inertial(hip_mesh, M_HIP))
        A(f'{S}  <geom class="visual" mesh="{hip_mesh[:-4]}"/>')
        A(f'{S}  <body name="{name}_upper_leg_link" pos="0 {ABD_Y*sy:.3f} 0">')
        A(f'{S}    <joint name="{name}_upper_leg_joint" axis="0 1 0" '
          f'range="{R_UPPER[0]:.4f} {R_UPPER[1]:.4f}"/>')
        A(f'{S}    ' + inertial(up_mesh, M_UPPER))
        A(f'{S}    <geom class="visual" mesh="{up_mesh[:-4]}"/>')
        if legs_collide:
            A(f'{S}    <geom name="{name}_thigh_col" class="collision" '
              f'type="capsule" fromto="0 0 -0.005 0 0 {-L_UPPER + 0.005:.3f}" '
              f'size="0.012"/>')
        A(f'{S}    <body name="{name}_lower_leg_link" pos="0 0 {-L_UPPER}">')
        A(f'{S}      <joint name="{name}_lower_leg_joint" axis="0 1 0" '
          f'range="{rl[0]:.4f} {rl[1]:.4f}"/>')
        A(f'{S}      ' + inertial("pp_vm_lowerleg_LR.stl", M_LOWER))
        A(f'{S}      <geom class="visual" mesh="pp_vm_lowerleg_LR"/>')
        if legs_collide:
            # 足球 (中心 -0.120 / 半径 0.012 -> 下端 -0.132) より 17mm 短く止める。
            # 4脚接地の姿勢で下腿カプセルが床に触れると接地センサの意味が変わる。
            A(f'{S}      <geom name="{name}_shank_col" class="collision" '
              f'type="capsule" fromto="0 0 -0.005 0 0 {-L_LOWER + 0.015:.3f}" '
              f'size="0.010"/>')
        A(f'{S}      <geom name="{name}_foot_col" class="collision" type="sphere" '
          f'size="{FOOT_R}" pos="0 0 {-L_LOWER}"/>')
        A(f'{S}      <site name="{name}_foot" pos="0 0 {-L_LOWER}" size="0.006" '
          'rgba="1 0.4 0.2 1"/>')
        A(f'{S}    </body>')
        A(f'{S}  </body>')
        A(f'{S}</body>')
    A('    </body>')
    A('  </worldbody>')
    A('')
    A('  <actuator>')
    for name, *_ in LEGS:
        for j, r in (("hip_joint", R_HIP), ("upper_leg_joint", R_UPPER),
                     ("lower_leg_joint", rl)):
            A(f'    <position class="pupper" name="{name}_{j}" joint="{name}_{j}" '
              f'ctrlrange="{r[0]:.4f} {r[1]:.4f}"/>')
    A('  </actuator>')
    A('')
    A('  <sensor>')
    A('    <!-- 実機 IMU で取得できる量 (MPU6050 等の追加が前提) -->')
    A('    <gyro name="gyro" site="imu"/>')
    A('    <accelerometer name="accelerometer" site="imu"/>')
    A('    <framezaxis name="upvector" objtype="site" objname="imu"/>')
    A('    <framequat name="orientation" objtype="site" objname="imu"/>')
    A('    <!-- 実機では取得できない特権情報 (critic 用 / 報酬用) -->')
    A('    <framelinvel name="global_linvel" objtype="site" objname="imu"/>')
    A('    <frameangvel name="global_angvel" objtype="site" objname="imu"/>')
    A('    <velocimeter name="local_linvel" site="imu"/>')
    for name, *_ in LEGS:
        A(f'    <framepos name="{name}_pos" objtype="site" objname="{name}_foot"/>')
        A(f'    <framelinvel name="{name}_linvel" objtype="site" objname="{name}_foot"/>')
    A('    <!-- 接触センサ: MJX で接触配列を走査せず済む -->')
    for name, *_ in LEGS:
        A(f'    <contact name="{name}_floor_found" geom1="{name}_foot_col" '
          'geom2="floor" reduce="mindist" num="1" data="found"/>')
    if terrain and legs_collide:
        # 胴体は箱 + 角球8 で表現してある。どれが触れても「胴体が床に着いた」。
        # geom ではなく body で引く (base_link の子 body は脚なので混ざらない)。
        A('    <contact name="trunk_floor_found" body1="base_link" geom2="floor" '
          'reduce="mindist" num="1" data="found"/>')
    else:
        A('    <contact name="trunk_floor_found" geom1="trunk_col" geom2="floor" '
          'reduce="mindist" num="1" data="found"/>')
    A('  </sensor>')
    A('')
    A('  <keyframe>')
    q = " ".join(f"{v:.4f}" for _, *_ in LEGS for v in (Q_HIP, Q_UPPER, Q_LOWER))
    A(f'    <key name="stand" qpos="0 0 {STAND_H + FOOT_R:.4f} 1 0 0 0 {q}" '
      f'ctrl="{q}"/>')
    A('  </keyframe>')
    A('</mujoco>')
    return "\n".join(x) + "\n"


if __name__ == "__main__":
    for fn, kw in (("pupper.xml", {}), ("pupper_terrain.xml", {"terrain": True}),
                   ("pupper_recover.xml", {"legs_collide": True}),
                   ("pupper_unified.xml", {"terrain": True, "legs_collide": True})):
        out = os.path.join(os.path.dirname(__file__), "..", "model", fn)
        open(out, "w").write(build(**kw))
        print("wrote", os.path.normpath(out))
