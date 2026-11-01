"""WSLg 上の対話ビューア。モデル・地形・ポリシーを目で確認するためのもの。

使い方:
  python scripts/view.py                       # 平地・基準姿勢で静止
  python scripts/view.py --gait                # 平地でスクリプト歩容を再生
  python scripts/view.py --policy out/train/t30/ckpt/latest.pkl        # 平地で方策
  python scripts/view.py --step 30  --policy out/train/t30/ckpt/latest.pkl  # 単一段差 30mm
  python scripts/view.py --stairs 30 --policy out/train/t30/ckpt/latest.pkl # 階段 30mm x6
  python scripts/view.py --terrain --policy out/train/t30/ckpt/latest.pkl   # 学習と同じ地形

--step / --stairs / --terrain のいずれかを付けると model/pupper_terrain.xml
(hfield) を読み、env/terrain.py と同じ高さ場を書き込む。--step は
scripts/step_test.py のベンチと同じ「x=0 に 1段だけ」の設定なので、
step_test の数字と目で見た挙動を突き合わせられる。

描画について: 学習用 XML の hfield (16m 角 / 641^2 格子) をそのまま描くと WSLg で
4 FPS しか出ない。ビューアでは必要な範囲だけを同じセル寸法 25mm で切り出し
(--extent)、影と反射を既定で切っている (--shadows で戻せる)。

キー操作 (ビューアのウィンドウにフォーカスがある状態で):
  W / S   前後に並進 (--vx, 既定 0.22 m/s)
  A / D   左右に並進 (--vy, 既定 0.15 m/s)
  ← / →   左右に旋回 (--wz, 既定 0.6 rad/s)
  P       スポーン位置へ置き直し        T   追従カメラの ON/OFF
押している間だけ指令が出て、離すと 0 に戻る。無人で段差に当てたいときは --cruise。
指令が効くのは --policy を付けたときだけ (--gait はスクリプト歩容で指令を見ない)。

キーは X サーバから直接読み (XQueryKeymap)、使うキーは XGrabKey で横取りして
ビューアに渡さない。ビューア組み込みの表示フラグのショートカット (W=Wireframe
など英字26文字すべて) はそのぶん効かなくなる。ビューアのウィンドウにフォーカス
を当てて押すこと。X11 が無い環境では押すたびに増減する方式に落ちる。

学習を止めずに使えるよう JAX は CPU に固定してある (ckpt の読み込みにしか
使わない)。それでも WSLg の描画で GPU は少し食うので、確認時だけにすること。
"""
import argparse, os, sys, time

# 学習中でも GPU メモリを奪わないこと。ckpt (.pkl) の中身は jax 配列なので
# pickle.load が jax を引っ張ってくる。import より前に固定する必要がある。
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np, mujoco, mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from gait_demo import ik, foot_traj, LEGS, PHASE, COM_X
from env import constants as consts
from env.deploy import (ObsBuilder, DEFAULT_POSE, CTRL_DT, OBS_DIM,
                        ACTION_SCALE, RECOVER_ACTION_SCALE, RECOVER_PRONE_DELTA,
                        balance_pose, balance_command, BALANCE_LIFT)

ap = argparse.ArgumentParser()
ap.add_argument("--gait", action="store_true", help="スクリプト歩容を再生")
ap.add_argument("--policy", default=None, help="学習済み方策 (.pkl の ckpt / .onnx)")
# --- 地形 ---
ap.add_argument("--step", type=float, default=None,
                help="x=0 に高さ [mm] の段差を1段だけ置く (step_test.py と同じ設定)")
ap.add_argument("--stairs", type=float, default=None, help="階段 1段の高さ [mm]")
ap.add_argument("--tread", type=float, default=0.30, help="階段の踏面 [m]")
ap.add_argument("--nsteps", type=int, default=6, help="階段の段数")
ap.add_argument("--terrain", action="store_true", help="学習と同じランダム地形")
ap.add_argument("--recover", action="store_true",
                help="転倒復帰モード (env/recover.py の方策用)。仰向けに置いて "
                     "起き上がりを見る。脚にも衝突形状のある XML を読み、"
                     "残差スケールを 1.6 に切り替える")
ap.add_argument("--roll", type=float, default=None,
                help="初期ロール角 [deg]。180=仰向け, 90=横倒し, 0=正立。"
                     "既定は --recover なら 180 (起き上がりを見るモードなので)、"
                     "--unified なら 0 (立った状態から歩かせ、転ばせたいときに "
                     "--roll 180 を付ける)")
ap.add_argument("--prone", action="store_true",
                help="--recover で「伏せ」(前脚を投げ出した低い姿勢) から始める。"
                     "--roll は 0 になる。仰向けからの復帰はまだ完成していないが、"
                     "この姿勢からなら 87.5%% 立ち上がる (OUTPUTS.md 参照)")
ap.add_argument("--hold-secs", type=float, default=3.0,
                help="--recover で復帰に成功したあと何秒見てから置き直すか")
ap.add_argument("--give-up", type=float, default=15.0,
                help="--recover で何秒で復帰できなければ置き直すか")
ap.add_argument("--unified", action="store_true",
                help="段差歩行 + 転倒復帰の両立方策 (env/unified.py) を見る。"
                     "地形は --terrain / --step / --stairs で選べる (既定は "
                     "学習と同じランダム地形)。--roll / --prone を付けると "
                     "その姿勢で転がして置く (転んでも置き直さない)。"
                     "可動域は傾きで 0.25 <-> 1.6 rad に自動で切り替わる")
ap.add_argument("--balance", default=None, choices=["a", "b"],
                help="対角2脚での静止保持モード (env/balance.py の方策用)。"
                     "a=lf+rh で立つ, b=rf+lh で立つ")
ap.add_argument("--extent", type=float, default=None,
                help="地形を描く範囲の半辺 [m]。既定 3.0 (--step/--stairs) / "
                     "4.0 (--terrain)。広げるほど描画が重くなる")
ap.add_argument("--shadows", action="store_true",
                help="影と反射を描く (WSLg では 1枚あたり 140ms ほど重くなる)")
ap.add_argument("--difficulty", type=float, default=1.0, help="--terrain の難易度 0-1")
ap.add_argument("--seed", type=int, default=0, help="--terrain の地形シード")
ap.add_argument("--spawn", default=None, help="初期位置 'x,y' [m]")
# --- 指令 ---
# キーを押している間だけ出す指令の大きさ (離すと 0)
ap.add_argument("--vx", type=float, default=0.22, help="W/S で出す前後速度 [m/s]")
ap.add_argument("--vy", type=float, default=0.15, help="A/D で出す横速度 [m/s]")
ap.add_argument("--wz", type=float, default=0.6, help="←/→ で出す旋回速度 [rad/s]")
ap.add_argument("--cruise", action="store_true",
                help="W/S を押していない間も --vx で前進し続ける (無人で段差に当てる用)")
ap.add_argument("--gait-freq", type=float, default=1.5,
                help="観測の位相クロック [Hz]。学習は 1.0-2.0 をランダム化している")
ap.add_argument("--no-tanh", action="store_true",
                help=".onnx が既に tanh 済みの場合に付ける (下の注記を参照)")
ap.add_argument("--no-follow", action="store_true", help="追従カメラを使わない")
ap.add_argument("--no-auto-reset", action="store_true", help="転倒しても置き直さない")
# --- スクリプト歩容のパラメータ ---
ap.add_argument("--freq", type=float, default=2.0)
ap.add_argument("--stride", type=float, default=0.07)
ap.add_argument("--height", type=float, default=0.163)
ap.add_argument("--duty", type=float, default=0.5)
a = ap.parse_args()

USE_TERRAIN = (a.terrain or a.unified
               or a.step is not None or a.stairs is not None)

# --- 段差歩行 + 転倒復帰の両立モード ---
# 可動域・速度コマンド・歩容位相を、IMU の傾きだけで切り替える方策
# (env/unified.py)。ObsBuilder(unified=True) が実機と同じ規則で切り替える。
UNI = a.unified
# --roll の既定値はモードで変える。--recover は「仰向けに置いて起き上がりを
# 見る」のが目的なので 180、--unified は歩行が主なので 0 から始める
# (仰向けから見たいときは --roll 180 を明示する)。
if a.roll is None:
    a.roll = 0.0 if (UNI and not a.recover) else 180.0

# --- 対角2脚保持モード ---
# 基準姿勢が4脚接地ではなくトロット中間姿勢になり、速度コマンドの枠は
# 「どちらの対角で立つか」の指定に変わる (env/balance.py を参照)。
# --- 転倒復帰モード ---
# 基準姿勢は歩行と同じだが、残差スケールが 1.6 (歩行は 0.25)。0.25 では
# 仰向けの機体から足先が床に届かない (env/recover.py の冒頭を参照)。
REC = a.recover
if REC:
    a.gait_freq = 0.0          # 位相クロックは使わない (学習時と同じ定数入力)
if REC or UNI:
    if a.prone:
        a.roll = 0.0           # 伏せは正立のまま脚だけ畳んだ姿勢
if UNI and a.gait_freq > 1.2:
    # 両立版は e40 と同じ 0.7-1.2Hz で学習している。既定の 1.5 は範囲外。
    # 上端の 1.2 が段差に一番強い (40mm の踏破率 1.2Hz 70.8%% / 0.7Hz 4.2%%)。
    a.gait_freq = 1.2

BAL = a.balance
if BAL is not None:
    assert np.allclose(BALANCE_LIFT, consts.BALANCE_LIFT), \
        "env/deploy.py と env/constants.py の BALANCE_LIFT がずれている"
    BAL_POSE = balance_pose(BAL)
    BAL_STANCE = np.array(consts.BALANCE_PAIRS[BAL], bool)
    a.gait_freq = 0.0        # 位相クロックは使わない (学習時と同じ定数入力)


# ---------------------------------------------------------------- 地形
# 学習用 XML の hfield は 16m 角 x 641^2 格子 = 82万ポリゴンあり、そのまま描くと
# WSLg では 4 FPS しか出ない (実測 237 ms/frame)。ビューアでは見える範囲だけを
# 切り出して描く。セル寸法 (25mm) は据え置くこと: hfield は格子点を結んだ三角形
# なので、セルを粗くすると段差の角が斜面に化けて「登りやすい別物」になる。
HF_CELL = 2 * consts.TERRAIN_HALF / (consts.TERRAIN_N - 1)      # 0.025 m


def make_heightmap(half):
    """(N, N) の高さ [m]。行が y、列が x (MuJoCo hfield の並び)。"""
    n = int(round(2 * half / HF_CELL)) + 1
    x = -half + np.arange(n) * HF_CELL
    hm = np.zeros((n, n), np.float32)

    if a.step is not None:
        # step_test.py の StepBench と同じ「x=0 より先が一段高い」だけの地形。
        hm[:, x >= 0.0] = a.step / 1000.0
    elif a.stairs is not None:
        h = a.stairs / 1000.0
        lvl = np.clip(np.floor(x / a.tread) + 1, 0, a.nsteps)
        hm[:, :] = (np.where(x >= 0.0, lvl, 0.0) * h).astype(np.float32)[None, :]
    else:
        # 学習と同じ地形を作ってから中央を切り出す。タイルの配置が変わらない
        # ので、difficulty / seed が同じなら学習で見ている地形そのもの。
        from env.terrain import heightmap
        full = heightmap(a.difficulty, a.seed)
        c = consts.TERRAIN_N // 2
        k = n // 2
        hm = np.ascontiguousarray(full[c - k:c - k + n, c - k:c - k + n])

    return hm


def ground_height(hm, xy):
    """xy [m] の地面高さ [m]。双線形補間 (env/terrain.py の _ground_height と同じ)。"""
    n = hm.shape[0]
    f = np.clip((np.asarray(xy) + HF_HALF) / HF_CELL, 0.0, n - 1.001)
    c, r = int(f[0]), int(f[1])
    tx, ty = f[0] - c, f[1] - r
    return float((hm[r, c] * (1 - tx) + hm[r, c + 1] * tx) * (1 - ty) +
                 (hm[r + 1, c] * (1 - tx) + hm[r + 1, c + 1] * tx) * ty)


def ground_height_max(hm, xy):
    """近傍4点の最大値。置いた瞬間に地面へめり込ませないための安全側の評価。"""
    n = hm.shape[0]
    f = np.clip((np.asarray(xy) + HF_HALF) / HF_CELL, 0.0, n - 1.001)
    c, r = int(f[0]), int(f[1])
    return float(max(hm[r, c], hm[r, c + 1], hm[r + 1, c], hm[r + 1, c + 1]))


xml = (consts.UNIFIED_XML if UNI
       else consts.TERRAIN_XML if USE_TERRAIN
       else consts.RECOVER_XML if REC else consts.XML)
hmap = HF_HALF = None

if not USE_TERRAIN:
    m = mujoco.MjModel.from_xml_path(str(xml))
else:
    # 描く範囲。階段は段数ぶんの長さが要る。
    HF_HALF = a.extent if a.extent else (4.0 if a.terrain else 3.0)
    if a.stairs is not None:
        HF_HALF = max(HF_HALF, a.nsteps * a.tread + 1.5)
    # 格子点数から半辺を決め直す。ground_height が HF_CELL を前提にしているので、
    # セル寸法をここで丸めずに保つ。
    HF_HALF = (int(round(2 * HF_HALF / HF_CELL))) / 2 * HF_CELL
    hmap = make_heightmap(HF_HALF)
    n = hmap.shape[0]
    spec = mujoco.MjSpec.from_file(str(xml))
    hf = spec.hfields[0]
    hf.nrow, hf.ncol = n, n
    # 縦スケールは地形の最大高さに合わせる (env/terrain.py の set_hfield と同じ)。
    # XML の TERRAIN_ZMAX は初期値にすぎないので、階段を何段積んでも頭打ちしない。
    ZMAX = max(float(hmap.max()), 1e-3)
    hf.size = [HF_HALF, HF_HALF, ZMAX, 0.5]
    hf.userdata = np.zeros(n * n)
    m = spec.compile()
    # MuJoCo は hfield_data * hfield_size[2] を高さとして解釈する。
    m.hfield_data[:] = (hmap / ZMAX).ravel()

if not a.shadows:
    # 影と反射は 1枚あたり 140 ms 効く。既定で切り、必要なら --shadows。
    m.light_castshadow[:] = 0
    m.mat_reflectance[:] = 0

d = mujoco.MjData(m)

# 段差の手前 0.8m に、段に正対させて置く (step_test.py の SPAWN_X と同じ)
if a.spawn is not None:
    SPAWN_XY = np.array([float(v) for v in a.spawn.split(",")], np.float64)
elif a.step is not None or a.stairs is not None:
    SPAWN_XY = np.array([-0.8, 0.0])
else:
    SPAWN_XY = np.array([0.0, 0.0])           # --terrain の中央は平地


# ---------------------------------------------------------------- 方策
# scripts/policy.py の NumpyPolicy をそのまま使う。以前ここに同じものを
# 手で持っていたが、隠れ層の活性化を tanh にしていて (brax の既定は silu)、
# 学習済みの方策が別物になっていた。実装は1か所に置くこと。
from policy import NumpyPolicy


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, obs):
        y = self.sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        # scripts/export_onnx.py は tanh を含めずに平均をそのまま出している。
        # 実機と同じ値を見たいときだけ --no-tanh で素通しにする。
        return y[:12] if a.no_tanh else np.tanh(y[:12])


policy = obs_builder = None
if a.policy:
    policy = (NumpyPolicy(a.policy, obs_dim=OBS_DIM) if a.policy.endswith(".pkl")
              else OnnxPolicy(a.policy))
    obs_builder = ObsBuilder(
        gait_freq=a.gait_freq, nominal_pose=BAL_POSE if BAL else None,
        action_scale=RECOVER_ACTION_SCALE if REC else ACTION_SCALE,
        unified=UNI)

act = {f"{l}_{j}_joint": m.actuator(f"{l}_{j}_joint").id
       for l in LEGS for j in ("hip", "upper_leg", "lower_leg")}
TORSO = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, consts.ROOT_BODY)


def sensor_slice(name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return slice(m.sensor_adr[sid], m.sensor_adr[sid] + m.sensor_dim[sid])


S_GYRO = sensor_slice(consts.GYRO_SENSOR)
S_ACCEL = sensor_slice(consts.ACCELEROMETER_SENSOR)
S_UP = sensor_slice(consts.UPVECTOR_SENSOR)
S_VEL = sensor_slice(consts.LOCAL_LINVEL_SENSOR)
FEET = np.array([m.site(s).id for s in consts.FEET_SITES])


def foot_clearance():
    """4足の足先クリアランス [m] (足球の下端が床から何m浮いているか)。"""
    z = d.site_xpos[FEET][:, 2] - consts.FOOT_RADIUS
    if hmap is not None:
        z = z - np.array([ground_height(hmap, p) for p in d.site_xpos[FEET][:, :2]])
    return z


def reset():
    # mj_resetDataKeyframe は d.time も 0 に戻す。時刻を基準にしている表示用
    # タイマと転倒判定を一緒に巻き戻さないと、次のエピソードの数秒間だけ
    # ステータス行が出なくなる。
    global t_print, fallen_since, bal_done, rec_done, rec_t, rec_best
    t_print, fallen_since, bal_done = 0.0, None, False
    rec_done, rec_t, rec_best = False, None, -1.0
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0:2] = SPAWN_XY
    if hmap is not None:
        d.qpos[2] = (ground_height_max(hmap, SPAWN_XY)
                     + consts.STAND_HEIGHT + consts.FOOT_RADIUS + 0.005)
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]          # yaw = 0 (+x を向く)
    d.qvel[:] = 0.0
    d.ctrl[:] = DEFAULT_POSE
    if REC or (UNI and (abs(a.roll) > 1e-6 or a.prone)):
        # ロール角を付けて床の上に置き直す (env/recover.py の reset と同じ手順)。
        # 接触する可能性があるのは足球・脚カプセル・胴体箱。最下点が床に
        # 触れる高さまで落とす。
        th = np.radians(a.roll) / 2
        d.qpos[3:7] = [np.cos(th), np.sin(th), 0.0, 0.0]
        # 伏せ姿勢は env/recover.py の prone_delta と同じ値を使う
        # (学習時の初期姿勢のひとつ)。
        d.qpos[7:] = DEFAULT_POSE + (
            np.tile(RECOVER_PRONE_DELTA, 4) if a.prone else 0.0)
        d.qpos[2] = 0.4
        mujoco.mj_forward(m, d)
        lows = []
        for gi in range(m.ngeom):
            if m.geom_conaffinity[gi] != 1:
                continue
            z, R, t = d.geom_xpos[gi][2], d.geom_xmat[gi].reshape(3, 3), m.geom_type[gi]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                lows.append(z - m.geom_size[gi][0])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                lows.append(z - abs(R[2, 2] * m.geom_size[gi][1]) - m.geom_size[gi][0])
            else:
                lows.append(z - np.abs(R[2] * m.geom_size[gi]).sum())
        # 地形の上では「その場の地面の高さ」から測る。
        gh = ground_height_max(hmap, SPAWN_XY) if hmap is not None else 0.0
        d.qpos[2] -= min(lows) - gh - 0.002
    if BAL is not None:
        # トロット中間姿勢で、接地脚がちょうど床に触れる高さに置く
        # (env/balance.py の reset と同じ手順)。
        d.qpos[7:] = BAL_POSE
        d.ctrl[:] = BAL_POSE
        mujoco.mj_forward(m, d)
        d.qpos[2] -= foot_clearance()[BAL_STANCE].min() - 0.001
    mujoco.mj_forward(m, d)
    if obs_builder is not None:
        obs_builder.reset()


# 指令は学習時のサンプリング範囲で頭打ちにする。範囲外を入れると方策が
# 見たことのない観測になり、「段差を越えられない」のか「指令が無茶」なのかが
# 切り分けられなくなる。
CMD_LIM = np.array([consts.CMD_VX, consts.CMD_VY, consts.CMD_WZ], np.float32)
cmd = np.zeros(3, np.float32)
if BAL is not None:
    cmd = balance_command(BAL)        # [±1, 0, 0]: 速度指令ではなく対角ペア
follow = not a.no_follow
want_reset = False


# キーを押している間だけ出す指令の大きさ。離せば 0 に戻る (--cruise 時は vx 据置)。
MAG = np.clip(np.abs([a.vx, a.vy, a.wz]), 0.0, CMD_LIM[:, 1]).astype(np.float32)

V = None                 # ビューアのハンドル (with ブロックで入る)
opt_flags_backup = None  # 直前フレームの可視化フラグ


class X11Keys:
    """X サーバから「今押されているキー」を直接読む。

    MuJoCo の passive viewer 経由では「押している間だけ」が作れない:
      - simulate/glfw_adapter.cc の IsKeyDownEvent が act == GLFW_PRESS なので、
        離したイベントもオートリピートも Python 側に来ない。
    さらにビューアは英字キーを表示フラグのショートカットに使っており
    (mujoco.mjVISSTRING / mjRNDSTRING)、UI パネルを隠しても src/ui/ui_main.c の
    ショートカット走査は止まらない。描画フラグ側 (W=Wireframe) は passive
    viewer から触れないので、押された後に戻すこともできない。

    そこで XGrabKey で使うキーだけを横取りし、ビューアには渡さない
    (= 既定ショートカットの無効化)。状態の読み出しは XQueryKeymap で、
    オートリピートの影響を受けない物理状態をそのまま見る。

    キーは X サーバに届いたものしか見えない。ビューアのウィンドウに
    フォーカスを当てて操作すること (Windows 側の端末に当てていると届かない)。
    """

    _KEYSYMS = {"W": 0x077, "A": 0x061, "S": 0x073, "D": 0x064,
                "LEFT": 0xff51, "RIGHT": 0xff53, "P": 0x070, "T": 0x074}
    _ANY_MODIFIER = 1 << 15
    _GRAB_ASYNC = 1

    def __init__(self):
        import ctypes, ctypes.util
        self.ct = ctypes
        path = ctypes.util.find_library("X11")
        if not path:
            raise RuntimeError("libX11 が見つからない")
        self.x = ctypes.CDLL(path)
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.dpy = self.x.XOpenDisplay(None)
        if not self.dpy:
            raise RuntimeError(f"X ディスプレイを開けない (DISPLAY={os.environ.get('DISPLAY')})")
        self.dpy = ctypes.c_void_p(self.dpy)
        # 他のクライアントが同じキーを掴んでいると BadAccess で落ちる。無視する。
        self._handler = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda d, e: 0)
        self.x.XSetErrorHandler(self._handler)
        self.x.XDefaultRootWindow.restype = ctypes.c_ulong
        self.root = self.x.XDefaultRootWindow(self.dpy)
        self.codes = {}
        for name, sym in self._KEYSYMS.items():
            kc = self.x.XKeysymToKeycode(self.dpy, ctypes.c_ulong(sym))
            if not kc:
                continue
            self.codes[name] = kc
            self.x.XGrabKey(self.dpy, kc, self._ANY_MODIFIER, ctypes.c_ulong(self.root),
                            False, self._GRAB_ASYNC, self._GRAB_ASYNC)
        self.x.XSync(self.dpy, False)
        self._buf = (ctypes.c_char * 32)()

    def pressed(self):
        """今押されているキー名の集合。"""
        self.x.XQueryKeymap(self.dpy, self._buf)
        out = set()
        for name, kc in self.codes.items():
            if self._buf[kc >> 3][0] & (1 << (kc & 7)):
                out.add(name)
        return out

    def close(self):
        for kc in self.codes.values():
            self.x.XUngrabKey(self.dpy, kc, self._ANY_MODIFIER,
                              self.ct.c_ulong(self.root))
        self.x.XCloseDisplay(self.dpy)


def key_callback(keycode):
    """X11 が使えないときの代替。押した瞬間しか来ないので増減して保持する。

    この経路では英字キーがビューア組み込みの表示フラグも切り替えてしまう。
    mjVISSTRING 側 (MjvOption.flags) は直前フレームの値に戻せるが、
    mjRNDSTRING 側 (W=Wireframe など) は passive viewer から触れないので戻せない。
    """
    global want_reset, follow
    k = chr(keycode) if 0 <= keycode < 0x110000 else ""
    if k == "W": bump(0, +0.05)
    elif k == "S": bump(0, -0.05)
    elif k == "A": bump(1, +0.05)
    elif k == "D": bump(1, -0.05)
    elif keycode == 263: bump(2, +0.2)        # GLFW_KEY_LEFT
    elif keycode == 262: bump(2, -0.2)        # GLFW_KEY_RIGHT
    elif k == "P": want_reset = True
    elif k == "T": follow = not follow
    else:
        return                        # 制御キー以外はビューアに任せる
    if V is not None and opt_flags_backup is not None:
        V.opt.flags[:] = opt_flags_backup     # 巻き添えで変わった表示を戻す
    print(f"\n指令 vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}"
          f"{'  (方策なしのため動きません)' if policy is None else ''}")


def bump(i, dv):
    if BAL is not None or REC:
        # balance は command 枠が対角ペアの指定。復帰は [0,0,0] 固定で学習して
        # いるので、ここを動かすと方策が学習時に見たことのない観測になる。
        return
    cmd[i] = np.clip(cmd[i] + dv, CMD_LIM[i, 0], CMD_LIM[i, 1])


def apply_held(held):
    """押されているキーから指令を作る。離したぶんは 0 に戻る。"""
    if BAL is not None or REC:
        return        # 上の bump() と同じ理由
    cmd[0] = MAG[0] * (("W" in held) - ("S" in held))
    cmd[1] = MAG[1] * (("A" in held) - ("D" in held))
    cmd[2] = MAG[2] * (("LEFT" in held) - ("RIGHT" in held))
    if a.cruise and not ("W" in held or "S" in held):
        cmd[0] = MAG[0]


reset()
n_sub = max(1, int(round(CTRL_DT / m.opt.timestep)))     # 方策は 50 Hz

keys = None
try:
    keys = X11Keys()
    apply_held(set())
except Exception as e:                    # Wayland のみ等。押すたびの増減に落とす
    print(f"X11 から直接キーを読めません ({e})。"
          "\n  押している間だけの操作はできないので、押すたびに増減する方式に"
          "切り替えます (ビューアの表示ショートカットとも干渉します)")
    if BAL is None:
        cmd[:] = np.clip([a.vx if a.cruise else 0.0, 0.0, 0.0],
                         CMD_LIM[:, 0], CMD_LIM[:, 1])
if USE_TERRAIN:
    what = (f"単一段差 {a.step:.0f}mm" if a.step is not None else
            f"階段 {a.stairs:.0f}mm x{a.nsteps} (踏面 {a.tread}m)" if a.stairs is not None
            else f"ランダム地形 difficulty={a.difficulty} seed={a.seed}")
    print(f"地形: {what}  最大高さ {hmap.max()*1000:.0f}mm  "
          f"描画範囲 ±{HF_HALF:.1f}m ({hmap.shape[0]}^2 格子, セル "
          f"{HF_CELL*1000:.0f}mm)"
          f"{'' if a.shadows else '  影/反射OFF'}")
print(f"方策: {a.policy or ('スクリプト歩容' if a.gait else 'なし (静止)')}"
      f"{'' if policy is None else f'  {n_sub} 物理ステップ/推論'}")
if REC:
    what = ("伏せ (前脚を投げ出した低い姿勢)" if a.prone else
            "仰向け" if a.roll > 135 else
            "横倒し" if a.roll > 45 else "正立")
    print(f"転倒復帰モード: {what} (ロール {a.roll:.0f}度) で置いて、"
          f"起き上がれるかを見る。残差スケール {RECOVER_ACTION_SCALE} "
          f"(歩行は {ACTION_SCALE})")
    print(f"判定: 水平度>0.85 かつ 胴体高 {consts.STAND_HEIGHT*1000:.0f}±30mm かつ"
          f" ほぼ静止。満たしたら {a.hold_secs:.0f}秒見てから置き直し、"
          f"{a.give_up:.0f}秒で満たせなければそこで置き直す"
          f"{' (--no-auto-reset で置き直さない)' if not a.no_auto_reset else ''}")
    print("操作: P=置き直し  T=追従カメラ  "
          "(W/S/A/D と ←/→ の速度指令はこのモードでは効きません。"
          "復帰方策は指令 [0,0,0] 固定で学習しています)")
    if policy is None:
        print("※ --policy を付けていないので、仰向けのまま動きません")
if UNI:
    what = ("伏せ" if a.prone else "仰向け" if a.roll > 135 else
            "横倒し" if a.roll > 45 else "正立")
    print(f"両立モード (段差歩行 + 転倒復帰): {what} (ロール {a.roll:.0f}度) から開始。"
          f"**転んでも置き直さない** (起き上がるのが仕事)")
    print(f"可動域は傾きで自動切替: 正立 ±{ACTION_SCALE}rad <-> 転倒 "
          f"±{RECOVER_ACTION_SCALE}rad (up {consts.GATE_UP_LO}-{consts.GATE_UP_HI} で補間)。"
          f"転倒中は速度指令を無視し歩容位相を止める")
    print("仰向けから試すなら --roll 180、伏せからなら --prone。"
          "P を押せばいつでも置き直せる")
    if policy is None:
        print("※ --policy を付けていないので、サーボは基準姿勢を保持するだけです")
if BAL is not None:
    clr0 = foot_clearance() * 1000
    print(f"対角2脚保持モード: {'lf+rh' if BAL == 'a' else 'rf+lh'} で立つ。"
          f"遊脚が床に触れたらそこで1回終了し、保持時間を出して置き直す")
    print(f"基準姿勢 (トロット中間姿勢) の遊脚クリアランス "
          f"{clr0[~BAL_STANCE][0]:.0f}/{clr0[~BAL_STANCE][1]:.0f}mm。"
          f"速度指令キーはこのモードでは効かない (command 枠は対角ペアの指定)")
    print("P:置き直し  T:追従カメラ")
    if policy is None:
        print("※ --policy を付けていないので、サーボは基準姿勢を保持するだけです")
elif not REC:
    hold = "押している間だけ" if keys is not None else "押すたびに増減 (X11 不可)"
    print(f"W/S:前後 ±{MAG[0]:.2f}m/s  A/D:左右 ±{MAG[1]:.2f}m/s  "
          f"←/→:旋回 ±{MAG[2]:.2f}rad/s  ({hold})")
    print("P:置き直し  T:追従カメラ"
          f"{'  ※ キーはビューアのウィンドウにフォーカスを当てて押すこと' if keys else ''}")
    if a.cruise:
        print(f"--cruise: W/S を押していない間も vx={MAG[0]:.2f} で前進します")
    if policy is None:
        print("※ --policy を付けていないので、指令を変えても機体は動きません"
              " (--gait はスクリプト歩容で指令を見ません)")
print("ビューアを起動します。ウィンドウが出ない場合は WSLg を確認してください。")

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as v:
    V = v
    v.opt.geomgroup[2] = 1
    v.opt.geomgroup[3] = 0
    v.cam.distance = 1.4
    v.cam.elevation = -20
    v.cam.azimuth = 135
    t_print = 0.0
    fallen_since = None
    falls = 0
    holds = []
    bal_done = False
    rec_done, rec_t, rec_best = False, None, -1.0
    rec_times = []
    prev_held = set()
    while v.is_running():
        loop_start = time.time()
        if keys is not None:
            held = keys.pressed()
            apply_held(held)
            for k in held - prev_held:        # 押した瞬間だけの操作
                if k == "P": want_reset = True
                elif k == "T": follow = not follow
            prev_held = held
        if want_reset:
            reset(); want_reset = False

        if policy is not None:
            gravity = -d.sensordata[S_UP]                 # 胴体座標系の重力方向
            obs = obs_builder.step(d.sensordata[S_GYRO], d.sensordata[S_ACCEL],
                                   gravity, cmd)
            d.ctrl[:] = obs_builder.commit(policy(obs))

        for _ in range(n_sub):
            if a.gait:
                t = d.time
                g = min(1.0, t / 1.0)
                for l in LEGS:
                    p = (t * a.freq + PHASE[l]) % 1.0
                    fx, fz = foot_traj(p, a.stride * g, 0.055 * g, a.duty, a.height)
                    th1, th2 = ik(fx + COM_X, fz)
                    d.ctrl[act[f"{l}_hip_joint"]] = 0.0
                    d.ctrl[act[f"{l}_upper_leg_joint"]] = th1
                    d.ctrl[act[f"{l}_lower_leg_joint"]] = th2
            mujoco.mj_step(m, d)

        # 制御キーの巻き添えで変わった表示を戻すための控え。ユーザが
        # 意図して切り替えた表示フラグはここで拾われるので残る。
        opt_flags_backup = v.opt.flags.copy()

        if follow:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = TORSO
        elif v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.sync()

        upright = d.sensordata[S_UP][2]

        # 対角2脚保持: 遊脚が触れた時点で1試行終了。保持時間を出して置き直す。
        if BAL is not None and not bal_done:
            clr = foot_clearance()
            if (clr[~BAL_STANCE] < 0.001).any() or upright < 0.2:
                holds.append(d.time)
                print(f"\n保持 {d.time:5.2f}s  ({len(holds)}回目, "
                      f"平均 {np.mean(holds):.2f}s / 最長 {max(holds):.2f}s)")
                if a.no_auto_reset:
                    bal_done = True          # 置き直さない場合は二重計上しない
                else:
                    reset()
                    continue

        # 地形の外に出ると足場が無く落ち続ける (姿勢は崩れないので転倒検知には
        # かからない)。env/terrain.py の _get_termination と同じ扱いで打ち切る。
        if hmap is not None and not a.no_auto_reset:
            if np.max(np.abs(d.qpos[:2])) > HF_HALF - 0.3:
                print(f"\n地形の外 (±{HF_HALF:.1f}m) に出た -> 置き直し"
                      "  (--extent で広げられます)")
                reset()
                continue

        # 転倒復帰: 立ち上がって静止したら成功。少し待って置き直す。
        if REC:
            h = d.qpos[2] - consts.FOOT_RADIUS
            rec_best = max(rec_best, float(upright))
            if (not rec_done and upright > 0.85
                    and abs(h - consts.STAND_HEIGHT) < 0.03
                    and np.linalg.norm(d.qvel[:3]) < 0.10):
                rec_done, rec_t = True, d.time
                rec_times.append(d.time)
                print(f"\n復帰 {d.time:5.2f}s  ({len(rec_times)}回目, "
                      f"平均 {np.mean(rec_times):.2f}s / 最速 {min(rec_times):.2f}s)")
            if not a.no_auto_reset and (
                    (rec_t is not None and d.time - rec_t > a.hold_secs)
                    or d.time > a.give_up):
                if rec_t is None:
                    print(f"\n{a.give_up:.0f}秒で復帰できず (最大水平度 "
                          f"{rec_best:+.2f}) -> 置き直し")
                reset()
                continue

        # 転倒検知。upvector の z が下を向いたら姿勢が崩れている。
        if BAL is None and not REC and not UNI and upright < 0.2:
            fallen_since = fallen_since or d.time
            if not a.no_auto_reset and d.time - fallen_since > 1.0:
                falls += 1
                print(f"\n転倒 ({falls}回目) -> 置き直し")
                reset(); fallen_since = None
        else:
            fallen_since = None

        if d.time - t_print > 0.25:
            t_print = d.time
            x, y, z = d.qpos[:3]
            gh = ground_height(hmap, (x, y)) if hmap is not None else 0.0
            vx = d.sensordata[S_VEL][0]
            if REC:
                print(f"\r t={d.time:5.2f}s  水平度 {upright:+.3f}  "
                      f"胴体高 {(z-consts.FOOT_RADIUS)*1000:4.0f}mm  "
                      f"接地脚 {int((d.sensordata[[m.sensor(s2).adr[0] for s2 in consts.FEET_CONTACT_SENSOR]] > 0).sum())}"
                      f"  {'復帰済み' if rec_done else '復帰中  '}   ",
                      end="", flush=True)
            elif BAL is not None:
                clr = foot_clearance() * 1000
                print(f"\r 保持 {d.time:5.2f}s  遊脚 "
                      f"[{' '.join(f'{v:4.0f}' for v in clr[~BAL_STANCE])}]mm"
                      f"  接地脚 [{' '.join(f'{v:4.0f}' for v in clr[BAL_STANCE])}]mm"
                      f"  胴体高 {(z-gh)*1000:4.0f}mm  水平度 {upright:+.3f}"
                      f"  ずれ {np.linalg.norm([x, y] - SPAWN_XY)*1000:4.0f}mm  ",
                      end="", flush=True)
            elif UNI:
                sc = obs_builder._scale if obs_builder is not None else 0.0
                print(f"\r t={d.time:6.1f}s  指令[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]"
                      f"  実速 vx={vx:+.2f}m/s  水平度 {upright:+.2f}"
                      f"  可動域 ±{sc:.2f}rad {'(歩行)' if sc < 0.4 else '(復帰)'}"
                      f"  胴体高 {(z-gh)*1000:4.0f}mm"
                      f"  位置 x={x:+.2f} y={y:+.2f}   ", end="", flush=True)
            else:
                print(f"\r t={d.time:6.1f}s  指令[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}]"
                      f"  実速 vx={vx:+.2f}m/s  位置 x={x:+.2f} y={y:+.2f}"
                      f"  地面 {gh*1000:4.0f}mm  胴体高 {(z-gh)*1000:4.0f}mm"
                      f"  水平度 {upright:+.2f}   ", end="", flush=True)

        dt = n_sub * m.opt.timestep - (time.time() - loop_start)
        if dt > 0: time.sleep(dt)

if keys is not None:
    keys.close()          # 横取りしていたキーを X サーバに返す
print()
