"""両立方策の通しデモ: 仰向けで置く -> 起き上がる -> そのまま歩き出す。

scripts/recover_handoff.py の 1 方策版。あちらは復帰方策 (rec9) と歩行方策
(e40) を外から切り替えていたが、こちらは **切り替えが無い**。方策は 1 つで、
速度指令もデモの最初から最後まで出しっぱなしにする。

実機と同じ経路で動かす:
    env/deploy.py の ObsBuilder(unified=True) + scripts/policy.py の NumpyPolicy
ObsBuilder が IMU の重力方向だけを見て

    残差スケール   1.6 (転倒) <-> 0.25 (正立)
    速度コマンド   転倒中は 0 に落とす
    歩容位相       転倒中は止める

を切り替える。**これは学習側 (env/unified.py) と同じ規則**で、外部から
モードを与えるものは何も無い。つまりこのスクリプトがそのまま ROS2 ノードの
骨格になる (歩行と復帰でノードを分ける必要がない)。

使い方:
  python scripts/unified_demo.py --ckpt out/train/u1/ckpt/final.pkl
  python scripts/unified_demo.py --ckpt ... --difficulty 1.0   # 段差地形の上で
"""
import argparse, os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np, mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from env import constants as consts
from env.deploy import ObsBuilder, DEFAULT_POSE, CTRL_DT, OBS_DIM, unified_action_scale
from policy import NumpyPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/train/u1/ckpt/final.pkl")
ap.add_argument("--roll", type=float, default=180.0,
                help="初期ロール [deg]。180=仰向け, 90=横倒し, 0=正立")
ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--secs", type=float, default=20.0)
ap.add_argument("--vx", type=float, default=0.2, help="前進指令 [m/s] (最初から出す)")
ap.add_argument("--gait-freq", type=float, default=1.2,
                help="歩容位相 [Hz]。**段差の踏破率はここで大きく変わる**。実測 (40mm): 0.7Hz 4.2%% / 1.0Hz 25.0%% / 1.2Hz 70.8%%。学習は 0.7-1.2 を一様に引いているので 1.2 はその上端")
ap.add_argument("--difficulty", type=float, default=0.0,
                help="0=平地, 1.0=学習と同じ段差地形 (--step-height の段差)")
ap.add_argument("--step-height", type=float, default=0.040)
ap.add_argument("--step", type=float, default=None, metavar="MM",
                help="x=0 に高さ [mm] の段を 1 つだけ置き、その手前 (x=-0.8) に"
                     "仰向けで置く。**起き上がって段を越えるまで**が 1 本の"
                     "動画に入る (--difficulty の地形は中央が平地なので、"
                     "20 秒では段まで届かないことがある)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--cam", default="track", choices=["track", "side"],
                help="track=斜め後方から追従 (既定), side=真横から追従。"
                     "40mm の段は俯瞰だとほぼ見えないので、段を見せたいときは side")
ap.add_argument("--filmstrip", action="store_true",
                help="コマ送りの PNG も出す。復帰は速いので前半を密に取る")
ap.add_argument("--out", default=os.path.join(ROOT, "out", "eval", "unified_demo"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(consts.UNIFIED_XML))
if a.step is not None:
    # x >= 0 を一段上げるだけの地形。step_test.py のベンチと同じ形。
    from env.terrain import set_hfield
    import numpy as _np
    n = consts.TERRAIN_N
    cell = 2 * consts.TERRAIN_HALF / (n - 1)
    x = -consts.TERRAIN_HALF + _np.arange(n) * cell
    hm = _np.zeros((n, n), _np.float32)
    hm[:, x >= 0.0] = a.step / 1000.0
    set_hfield(m, hm)
    a.difficulty = 1.0            # ground() に hfield を見させる
elif a.difficulty > 0:
    from env.terrain import heightmap, set_hfield
    hm = heightmap(a.difficulty, a.seed, a.step_height)
    set_hfield(m, hm)
else:
    m.hfield_size[0, 2] = 1e-3
    m.hfield_data[:] = 0.0
d = mujoco.MjData(m)
S_UP = m.sensor_adr[m.sensor(consts.UPVECTOR_SENSOR).id]
S_GYRO = m.sensor_adr[m.sensor(consts.GYRO_SENSOR).id]
S_ACC = m.sensor_adr[m.sensor(consts.ACCELEROMETER_SENSOR).id]
S_VEL = m.sensor_adr[m.sensor(consts.LOCAL_LINVEL_SENSOR).id]   # 機体座標の速度
FEET = [m.sensor(s).adr[0] for s in consts.FEET_CONTACT_SENSOR]

pol = NumpyPolicy(a.ckpt, obs_dim=OBS_DIM)
obs_b = ObsBuilder(gait_freq=a.gait_freq, unified=True)


def ground(xy):
    """xy 直下の地面高さ [m] (hfield の近傍4点の最大)。"""
    if a.difficulty <= 0:
        return 0.0
    n = consts.TERRAIN_N
    cell = 2 * consts.TERRAIN_HALF / (n - 1)
    f = np.clip((np.asarray(xy) + consts.TERRAIN_HALF) / cell, 0.0, n - 1.001)
    c, r = int(f[0]), int(f[1])
    h = hm
    return float(max(h[r, c], h[r, c + 1], h[r + 1, c], h[r + 1, c + 1]))


def place():
    """--roll の姿勢で、地面に触れる高さまで落として置く。"""
    mujoco.mj_resetData(m, d)
    cr, sr = np.cos(np.radians(a.roll) / 2), np.sin(np.radians(a.roll) / 2)
    cy, sy = np.cos(np.radians(a.yaw) / 2), np.sin(np.radians(a.yaw) / 2)
    d.qpos[3:7] = [cy * cr, cy * sr, sy * sr, sy * cr]     # yaw * roll
    d.qpos[7:] = DEFAULT_POSE
    d.ctrl[:] = DEFAULT_POSE
    d.qpos[0] = -0.8 if a.step is not None else 0.0   # 段の手前に置く
    d.qpos[2] = 0.6
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
    d.qpos[2] -= min(lows) - ground(d.qpos[:2]) - 0.002
    mujoco.mj_forward(m, d)


place()
n_sub = max(1, int(round(CTRL_DT / m.opt.timestep)))
N = int(a.secs / CTRL_DT)
HOLD = int(0.3 / CTRL_DT)      # 0.3s 連続で満たせば復帰とみなす
hold, t_rec, x_rec = 0, None, None
frames, log = [], []
renderer = None if a.no_video else mujoco.Renderer(m, height=480, width=640)
if a.cam == "side":
    # 真横から機体を追う自由カメラ。段の高さが輪郭で見える。
    side_cam = mujoco.MjvCamera()
    side_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    side_cam.trackbodyid = m.body(consts.ROOT_BODY).id
    side_cam.azimuth, side_cam.elevation, side_cam.distance = 90.0, -8.0, 1.1
    side_cam.lookat[2] = 0.05
print(f"方策 {a.ckpt} (1つだけ。切り替えなし)")
print(f"初期ロール {a.roll:.0f}度  地形 "
      f"{'平地' if a.difficulty <= 0 else f'段差 {a.step_height*1000:.0f}mm'}  "
      f"前進指令 {a.vx} m/s を最初から出しっぱなし")

cmd = np.array([a.vx, 0.0, 0.0], np.float32)
fwd = 0.0
for k in range(N):
    t = k * CTRL_DT
    gravity = -d.sensordata[S_UP:S_UP + 3]
    up = float(d.sensordata[S_UP + 2])
    gh = ground(d.qpos[:2])
    h = float(d.qpos[2]) - consts.FOOT_RADIUS - gh
    n_feet = int((d.sensordata[FEET] > 0).sum())
    obs = obs_b.step(d.sensordata[S_GYRO:S_GYRO + 3],
                     d.sensordata[S_ACC:S_ACC + 3], gravity, cmd)
    d.ctrl[:] = obs_b.commit(pol(obs))
    # 「起き上がった」の判定。**復帰専用 run (rec*) の基準は使えない**:
    # あちらは速度コマンドが常に 0 なので「接地脚>=3 かつ ほぼ静止」を要求
    # できるが、このデモは最初から歩けと指示しているので、起き上がった直後に
    # 歩き出す (静止しない)。歩行方策の立ち方自体も接地脚 2/4 で、これは
    # e40 単体で測っても同じ (足を浮かせ気味に立つ)。ここでは
    # **胴体が起きて所定の高さに乗ったか**だけを見る。
    ok = up > 0.85 and abs(h - consts.STAND_HEIGHT) < 0.03
    hold = hold + 1 if ok else 0
    if t_rec is None and hold >= HOLD:
        t_rec, x_rec = t - 0.5, float(d.qpos[0])
        print(f"\n復帰 {t_rec:.2f}s  (ここから先も同じ方策のまま歩き出す)")
    for _ in range(n_sub):
        mujoco.mj_step(m, d)
    # **前進距離は機体前方で測る**。仰向けから転がって起きると機体の向きが
    # 変わるので、ワールド x の変位では「指令どおり前に進んだか」を測れない
    # (実際、復帰後にきちんと前進していても x は減ることがある)。
    # 機体座標の速度を積分する。
    fwd += float(d.sensordata[S_VEL]) * CTRL_DT
    log.append((t, up, h, fwd, unified_action_scale(gravity)))
    if renderer is not None and k % 2 == 0:
        renderer.update_scene(d, camera=(side_cam if a.cam == "side" else "track"))
        frames.append(renderer.render())
    if k % 25 == 0:
        sc = unified_action_scale(gravity)
        print(f"\r t={t:5.2f}s 水平度 {up:+.3f} 胴体高 {h*1000:4.0f}mm "
              f"接地脚 {n_feet} 可動域 ±{sc:.2f}rad 前進 {fwd:+.2f}m   ",
              end="", flush=True)
print()

log = np.array(log)
if t_rec is None:
    print(f"\n{a.secs:.0f}s 以内に復帰できなかった (最大水平度 {log[:, 1].max():+.3f})")
else:
    w = log[log[:, 0] >= t_rec + 0.5]
    if len(w) > 1:
        dist = w[-1, 3] - w[0, 3]
        secs = w[-1, 0] - w[0, 0]
        print(f"\n復帰 {t_rec:.2f}s  |  その後 {secs:.1f}s で機体前方に "
              f"{dist:+.2f} m 前進 (平均 {dist/max(secs,1e-9):+.3f} m/s, "
              f"指令 {a.vx:+.2f})")
        print(f"復帰後の水平度 最小 {w[:, 1].min():+.3f} "
              f"({'転倒せず' if w[:, 1].min() > 0.5 else '**また転んだ**'})")
        print(f"可動域: 復帰中 最大 ±{log[log[:, 0] < t_rec, 4].max():.2f}rad  "
              f"復帰後 最大 ±{w[:, 4].max():.2f}rad")
if renderer is not None and a.filmstrip:
    # コマ送り: 復帰 (最初の 2 秒) を密に、そのあとの歩行を粗く。
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = 1.0 / (CTRL_DT * 2)                      # frames は 2 step ごと
    ts = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0] + \
         list(np.linspace(4.0, a.secs - 0.5, 5))
    idx = [min(int(t * fps), len(frames) - 1) for t in ts]
    n = len(idx)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 5.0))
    for j, (i, t) in enumerate(zip(idx, ts)):
        b = ax.ravel()[j]
        b.imshow(frames[i]); b.set_xticks([]); b.set_yticks([])
        up_t = log[min(int(t / CTRL_DT), len(log) - 1), 1]
        b.set_title(f"t={t:.1f}s  up={up_t:+.2f}", fontsize=9)
    for j in range(n, ax.size):
        ax.ravel()[j].axis("off")
    fig.suptitle(f"{os.path.basename(os.path.dirname(os.path.dirname(a.ckpt)))}: "
                 f"supine -> stand up ({t_rec:.2f}s) -> walk"
                 f"{f' and climb a {a.step:.0f}mm step' if a.step is not None else ' on rough terrain' if a.difficulty > 0 else ''}", fontsize=11)
    fig.tight_layout()
    tag = (f"_step{a.step:.0f}mm" if a.step is not None
           else "_terrain" if a.difficulty > 0 else "")
    fp = os.path.join(a.out, f"filmstrip{tag}.png")
    fig.savefig(fp, dpi=110); plt.close(fig)
    print(f"-> {fp}")

if renderer is not None:
    import imageio
    renderer.close()
    tag2 = (f"_step{a.step:.0f}mm" if a.step is not None
            else "_terrain" if a.difficulty > 0 else "")
    p = os.path.join(a.out, f"unified_demo{tag2}.mp4")
    imageio.mimsave(p, frames, fps=int(1 / CTRL_DT / 2), quality=8)
    print(f"-> {p}")
