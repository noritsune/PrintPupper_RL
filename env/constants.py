"""PrintPupper の定数。URDF / gait.yaml / ハードウェアノードから抽出・検証済み。"""
from etils import epath
import numpy as np

ROOT = epath.Path(__file__).parent.parent
XML = ROOT / "model" / "pupper.xml"

ROOT_BODY = "base_link"
LEGS = ["lf", "rf", "lh", "rh"]                 # 左前, 右前, 左後, 右後
JOINT_ORDER = [f"{l}_{j}_joint" for l in LEGS
               for j in ("hip", "upper_leg", "lower_leg")]   # joints.yaml の順序

FEET_SITES = [f"{l}_foot" for l in LEGS]
FEET_GEOMS = [f"{l}_foot_col" for l in LEGS]
FEET_POS_SENSOR = [f"{l}_pos" for l in LEGS]
FEET_LINVEL_SENSOR = [f"{l}_linvel" for l in LEGS]
FEET_CONTACT_SENSOR = [f"{l}_floor_found" for l in LEGS]
TRUNK_CONTACT_SENSOR = "trunk_floor_found"

GYRO_SENSOR = "gyro"
ACCELEROMETER_SENSOR = "accelerometer"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"

# 基準姿勢: gait.yaml の nominal_height(0.1626) / com_x_translation(0.0071) と
# 一致することを scripts/validate_model.py で検証済み。
DEFAULT_POSE = np.array([0.0, np.pi / 4, -np.pi / 2] * 4, dtype=np.float32)
STAND_HEIGHT = 0.1626
FOOT_RADIUS = 0.012

# 速度コマンド範囲。
# gait.yaml の値は max_linear_velocity_x=0.2 / y=0.1 / angular_z=0.7 だが、
# これは CHAMP の歩容生成器に対する保守的な上限。DS3235 (3Nm) はスクリプト
# 歩容だけで 0.48 m/s 出るので、余力を活かして 1.5 倍の範囲で学習する。
# 学習した方策は範囲内の低速コマンドもそのまま追従できる。
# gait.yaml どおりに揃えたい場合はここを (0.2, 0.1, 0.7) に戻す。
CMD_VX = (-0.30, 0.30)
CMD_VY = (-0.15, 0.15)
CMD_WZ = (-1.00, 1.00)
GAIT_YAML_LIMITS = (0.20, 0.10, 0.70)   # 参考: CHAMP 側の上限

# 歩容 (gait.yaml)
STANCE_DURATION = 0.6
SWING_HEIGHT = 0.055
GAIT_FREQ = (1.0, 2.0)      # Hz。stance_duration 0.6 の trot 相当を中心に

# --- 対角2脚での静止保持 (env/balance.py) ---
# トロットの中間姿勢: 対角2脚 (lf+rh または rf+lh) で立ち、残り2脚は
# 基準姿勢から下の差分だけ引き上げる。足先クリアランスは 42mm になる。
# env/deploy.py にも同じ値の写しがある (実機側は numpy だけで動かすため)。
# 変えたら両方を直すこと。scripts/view.py が起動時に一致を検査する。
BALANCE_LIFT = [0.0, 0.40, -0.45]       # [hip, upper_leg, lower_leg] の差分 [rad]
BALANCE_PAIRS = {"a": [1, 0, 0, 1],     # lf + rh を接地
                 "b": [0, 1, 1, 0]}     # rf + lh を接地

# --- 転倒復帰 (env/recover.py) ---
# 大腿・下腿にも衝突形状を付けた XML。歩行版は足球と胴体箱しか衝突しないので
# 脚が床をすり抜ける。歩行では足先しか床に触れないので問題にならないが、
# 仰向けから起き上がる動作では脚で床を押すため、すり抜けたままだと実機で
# 成立しない動きを覚える。4脚接地の姿勢では脚カプセルは床に触れない
# (下腿カプセルの下端は足球より 12.6mm 高い) ので、接地判定の意味は変わらない。
RECOVER_XML = ROOT / "model" / "pupper_recover.xml"
# 仰向けの機体が床に接する高さ (機体フレーム z [m])。胴体箱の上面。
TRUNK_TOP = 0.057

# --- 段差歩行 + 転倒復帰の両立 (env/unified.py) ---
# 段差地形 (hfield) と脚の衝突形状の両方を持つ XML。歩行版の地形と復帰版の
# 脚カプセルをそのまま合わせたもので、どちらの課題の前提も壊さない。
UNIFIED_XML = ROOT / "model" / "pupper_unified.xml"
# 可動域の切り替え。歩行は基準姿勢 ± 0.25 rad で十分かつその精度が要るが、
# 復帰は ± 1.6 rad 無いと足が床に届かない (env/recover.py の実測表)。
# 一つの方策で両方やるので、**傾きに応じて滑らかに切り替える**。
# 上端 (正立側) は歩行版と、下端 (転倒側) は復帰版と数値的に完全に一致するので、
# e40 / rec9 どちらの重みもそれぞれの領域ではそのままの意味で通用する。
WALK_ACTION_SCALE = 0.25
RECOVER_ACTION_SCALE = 1.6
# 切り替えの帯 (upvector z)。0.90 = 傾き 26度, 0.50 = 60度。
# 段差 40mm を越えるときの胴体の傾きは実測で最大 20度程度なので、
# 歩行中にこの帯へ入ることはない。IMU の姿勢推定誤差 (最大 3度程度) でも
# 帯をまたがない幅を取ってある。
GATE_UP_LO = 0.50
GATE_UP_HI = 0.90

# --- 段差地形 (env/terrain.py と scripts/build_mjcf.py --terrain で共有) ---
# 地形は hfield 1枚。geom 名は平地版と同じ "floor" のままにすること。
# pupper.xml の接触センサが geom2="floor" で引いているので、名前を変えると
# 足の接地判定が全滅する (feet_air_time / clearance / slip が死ぬ)。
TERRAIN_XML = ROOT / "model" / "pupper_terrain.xml"
TERRAIN_HALF = 8.0      # hfield の半辺 [m]。16m 角
TERRAIN_N = 641         # 一辺の格子点数。格子間隔 = 2*HALF/(N-1) = 0.025 m
TERRAIN_ZMAX = 0.20     # hfield データ値 1.0 に対応する高さ [m]。XML に書く初期値。
                        # 実際の縦スケールは env/terrain.py の set_hfield() が地形の
                        # 最大高さに合わせて実行時に書き直す (hfield_size[0, 2]) ので、
                        # ここは地形の高さ上限ではない。高い段差や長い階段を置くのに
                        # XML を作り直す必要はない。
TERRAIN_TILE = 2.0      # 地形タイル一辺 [m]
TERRAIN_BORDER = 0.25   # 各タイル外周に残す平坦帯 [m]。隣接タイルとの連続性を保証
TERRAIN_FLAT_R = 1.5    # 中央の平地半径 [m]
TERRAIN_SPAWN_R = 3.0   # スポーン位置のばらつき半径 [m]

# difficulty=1.0 での寸法。機体は STAND_HEIGHT 0.1626 / SWING_HEIGHT 0.055 なので
# Go1 用の 0.05m 段差をそのまま持ってくると絶対に登れない。
STEP_HEIGHT = 0.030     # 段差 1 段の高さ [m]。既定値。run ごとに
                        # train.py --step-height で上書きする
TERRAIN_ROUGH = 0.020   # 不整地のランダム凹凸の振幅 [m]
