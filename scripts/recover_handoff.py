"""転倒復帰 -> 歩行 の引き渡しデモ。

仰向けで置いて復帰方策で起き上がり、立ち上がったら歩行方策 (e40) に
切り替えて歩き出す。実機と同じ経路 (env/deploy.py の ObsBuilder + NumpyPolicy)
で動かすので、そのまま ROS2 ノードに移せる制御の骨格になる。

引き渡しが成立する理由:
  - 観測 38 次元が両方で同一。
  - 基準姿勢が両方とも「4脚接地の基準姿勢」。復帰方策は最後に action≈0
    (= 基準姿勢) で静止するよう報酬を付けてあるので、切り替えた瞬間に
    歩行方策が想定する姿勢から始まる。
  - 違うのは残差スケールだけ (復帰 1.6 / 歩行 0.25) と、位相クロック
    (復帰は停止 / 歩行は回す)。切り替えでこの2つを差し替える。

使い方:
  python scripts/recover_handoff.py --recover-ckpt out/train/rec1/ckpt/final.pkl
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
from env.deploy import (ObsBuilder, DEFAULT_POSE, CTRL_DT, OBS_DIM,
                        ACTION_SCALE, RECOVER_ACTION_SCALE)
from policy import NumpyPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--recover-ckpt", default="out/train/rec1/ckpt/final.pkl")
ap.add_argument("--walk-ckpt", default="out/train/e40/ckpt/final.pkl")
ap.add_argument("--roll", type=float, default=180.0, help="初期ロール [deg]")
ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--secs", type=float, default=16.0)
ap.add_argument("--walk-secs", type=float, default=8.0, help="引き渡し後に歩く秒数")
ap.add_argument("--vx", type=float, default=0.2)
ap.add_argument("--gait-freq", type=float, default=0.9, help="歩行方策の位相クロック")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--out", default=os.path.join(ROOT, "out", "eval", "handoff"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(consts.RECOVER_XML))
d = mujoco.MjData(m)
S_UP = m.sensor_adr[m.sensor(consts.UPVECTOR_SENSOR).id]
S_GYRO = m.sensor_adr[m.sensor(consts.GYRO_SENSOR).id]
S_ACC = m.sensor_adr[m.sensor(consts.ACCELEROMETER_SENSOR).id]
FEET = [m.sensor(s).adr[0] for s in consts.FEET_CONTACT_SENSOR]

rec_pol = NumpyPolicy(a.recover_ckpt, obs_dim=OBS_DIM)
walk_pol = NumpyPolicy(a.walk_ckpt, obs_dim=OBS_DIM)
rec_obs = ObsBuilder(gait_freq=0.0, action_scale=RECOVER_ACTION_SCALE)
walk_obs = ObsBuilder(gait_freq=a.gait_freq, action_scale=ACTION_SCALE)


def place():
    """仰向け (--roll) に置いて、床に触れる高さまで落とす。"""
    mujoco.mj_resetData(m, d)
    cr, sr = np.cos(np.radians(a.roll) / 2), np.sin(np.radians(a.roll) / 2)
    cy, sy = np.cos(np.radians(a.yaw) / 2), np.sin(np.radians(a.yaw) / 2)
    d.qpos[3:7] = [cy * cr, cy * sr, sy * sr, sy * cr]     # yaw * roll
    d.qpos[7:] = DEFAULT_POSE
    d.ctrl[:] = DEFAULT_POSE
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
    d.qpos[2] -= min(lows) - 0.002
    mujoco.mj_forward(m, d)


place()
n_sub = max(1, int(round(CTRL_DT / m.opt.timestep)))
N = int(a.secs / CTRL_DT)
HOLD = int(0.5 / CTRL_DT)
mode, hold, t_rec, t_switch = "recover", 0, None, None
frames, log = [], []
renderer = None if a.no_video else mujoco.Renderer(m, height=480, width=640)
print(f"復帰 {a.recover_ckpt}\n歩行 {a.walk_ckpt}\n初期ロール {a.roll:.0f}度")

for k in range(N):
    t = k * CTRL_DT
    gravity = -d.sensordata[S_UP:S_UP + 3]
    up = d.sensordata[S_UP + 2]
    h = d.qpos[2] - consts.FOOT_RADIUS
    n_feet = int((d.sensordata[FEET] > 0).sum())
    if mode == "recover":
        obs = rec_obs.step(d.sensordata[S_GYRO:S_GYRO + 3],
                           d.sensordata[S_ACC:S_ACC + 3], gravity, np.zeros(3))
        d.ctrl[:] = rec_obs.commit(rec_pol(obs))
        ok = (up > 0.85 and abs(h - consts.STAND_HEIGHT) < 0.03 and n_feet >= 3
              and np.linalg.norm(d.qvel[:3]) < 0.10)
        hold = hold + 1 if ok else 0
        if hold >= HOLD:
            t_rec = t - 0.5
            t_switch = t
            mode = "walk"
            walk_obs.reset()
            print(f"\n復帰 {t_rec:.2f}s -> 歩行方策へ引き渡し (t={t:.2f}s)")
    else:
        cmd = np.array([a.vx, 0.0, 0.0], np.float32)
        obs = walk_obs.step(d.sensordata[S_GYRO:S_GYRO + 3],
                            d.sensordata[S_ACC:S_ACC + 3], gravity, cmd)
        d.ctrl[:] = walk_obs.commit(walk_pol(obs))
        if t - t_switch > a.walk_secs:
            break
    for _ in range(n_sub):
        mujoco.mj_step(m, d)
    log.append((t, float(up), float(d.qpos[2] - consts.FOOT_RADIUS),
                float(d.qpos[0]), mode == "walk"))
    if renderer is not None and k % 2 == 0:
        renderer.update_scene(d, camera="track")
        frames.append(renderer.render())
    if k % 25 == 0:
        print(f"\r t={t:5.2f}s [{mode:7s}] 水平度 {up:+.3f} 胴体高 "
              f"{h*1000:4.0f}mm 接地脚 {n_feet} x={d.qpos[0]:+.2f}m   ",
              end="", flush=True)
print()

log = np.array(log)
if t_rec is None:
    print(f"\n{a.secs:.0f}s 以内に復帰できなかった (最大水平度 {log[:,1].max():+.3f})")
else:
    w = log[log[:, 4] > 0]
    print(f"\n復帰 {t_rec:.2f}s  |  引き渡し後 {w[-1,0]-w[0,0]:.1f}s で "
          f"x {w[0,3]:+.2f} -> {w[-1,3]:+.2f} m ({w[-1,3]-w[0,3]:+.2f} m 前進)")
    print(f"歩行中の水平度 最小 {w[:,1].min():+.3f} "
          f"({'転倒せず' if w[:,1].min() > 0.5 else '**転倒した**'})")
if renderer is not None:
    import imageio
    renderer.close()
    p = os.path.join(a.out, "handoff.mp4")
    imageio.mimsave(p, frames, fps=int(1 / CTRL_DT / 2), quality=8)
    print(f"-> {p}")
