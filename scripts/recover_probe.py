"""復帰方策が「どこで止まっているか」を姿勢そのもので切り分ける。

recover_test.py は成功率を出すが、失敗しているときに **何をしているのか**は
分からない。水平度 +0.54 という数字だけ見ると「半分起き上がっている」に
見えるが、実際には胴体の前端を床に着けた逆立ちで、後脚は床から 100mm 浮いて
いた。この取り違えで重みを何度も振り直したので、失敗の中身は必ずここで見る。

出すもの:
  - 水平度の時間推移 (どこで止まったか)
  - 最終姿勢: roll / pitch / 胴体高 / 脚ごとの基準姿勢からの差と足先高さ
  - **実際に床に触れている geom の名前** (これが決定的だった)
  - 左右反転した方策を同じ初期状態で走らせた結果
    -> 反転しても同じところで止まるなら、機体でなく課題の構造の問題。
       畳まれる脚だけが入れ替わるなら方策の左右非対称。
  - out/eval/<run>_probe/stuck.png  最終姿勢の画像 (2方向 x 3本)

使い方:
  python scripts/recover_probe.py --run rec4
  python scripts/recover_probe.py --ckpt out/train/rec7/ckpt/latest.pkl --side
"""
import argparse, functools, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--run", default=None)
ap.add_argument("--episodes", type=int, default=32)
ap.add_argument("--side", action="store_true", help="横倒しから始める (既定は仰向け)")
ap.add_argument("--secs", type=float, default=None)
ap.add_argument("--contacts-per-env", type=int, default=16)
ap.add_argument("--no-mirror", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import jax, jax.numpy as jp, mujoco
from env import _compat; _compat.install()
import warp as wp
from mujoco_playground._src import wrapper
from env import constants as consts
from env.randomize import domain_randomize
from env.recover import PupperRecover, recover_config
from policy import load

if args.run is None and args.ckpt is None:
    args.run = sorted(os.listdir(os.path.join(ROOT, "out", "train")))[-1]
ckpt = args.ckpt or os.path.join(ROOT, "out", "train", args.run, "ckpt", "final.pkl")
if not os.path.exists(ckpt):
    ckpt = ckpt.replace("final.pkl", "latest.pkl")
OUT = args.out or os.path.join(ROOT, "out", "eval", (args.run or "ckpt") + "_probe")
os.makedirs(OUT, exist_ok=True)

B = args.episodes
cfg = recover_config()
cfg.recover.p_flip = 0.0 if args.side else 1.0
cfg.recover.p_side = 1.0 if args.side else 0.0
cfg.recover.p_prone = 0.0
cfg.recover.p_stuck = 0.0
if args.secs is not None:
    cfg.episode_length = int(round(args.secs / cfg.ctrl_dt))
cfg.naconmax = args.contacts_per_env * B
env = PupperRecover(config=cfg)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
_, inference, _ = load(ckpt, env=env)
T = cfg.episode_length
mm = env.mj_model
UP = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id] + 2
TORSO = mm.body(consts.ROOT_BODY).id

# ---- 左右反転の写像 ----
# 機体は y=0 について鏡像。hip の軸は x なので角度は符号反転、upper/lower の
# 軸は y なのでそのまま。脚は lf<->rf, lh<->rh。ジャイロは擬ベクトルなので
# (wx, wy, wz) -> (-wx, wy, -wz)、重力・加速度は (gx, gy, gz) -> (gx, -gy, gz)。
JIDX = jp.array(np.concatenate([np.arange(3) + 3 * p for p in (1, 0, 3, 2)]))
JSGN = jp.array(np.tile([-1.0, 1.0, 1.0], 4))
mirror_act = lambda a: a[..., JIDX] * JSGN
def mirror_obs(s):
    return jp.concatenate([
        s[..., 0:3] * jp.array([-1.0, 1.0, -1.0]),
        s[..., 3:6] * jp.array([1.0, -1.0, 1.0]),
        s[..., 6:9] * jp.array([1.0, -1.0, 1.0]),
        s[..., 9:12] * jp.array([1.0, -1.0, -1.0]),
        mirror_act(s[..., 12:24]), mirror_act(s[..., 24:36]), s[..., 36:38]], -1)

rand_fn = functools.partial(domain_randomize,
                            rng=jax.random.split(jax.random.PRNGKey(42), B))
wrapped = wrapper.wrap_for_brax_training(
    env, episode_length=T, action_repeat=1, randomization_fn=rand_fn, full_reset=True)
w_reset, w_step = jax.jit(wrapped.reset), jax.jit(wrapped.step)

print(f"[probe] {ckpt}\n[start] {'横倒し' if args.side else '仰向け'} x {B} 本, "
      f"{T*env.dt:.0f}s\n")

def run(mirrored):
    state = w_reset(jax.random.split(jax.random.PRNGKey(0), B))
    ups = []
    for k in range(T - 1):
        obs = dict(state.obs)
        if mirrored:
            obs["state"] = mirror_obs(obs["state"])
        act, _ = inference(obs, jax.random.PRNGKey(k))
        if mirrored:
            act = mirror_act(act)
        state = w_step(state, act)
        ups.append(np.array(state.data.sensordata[:, UP]))
    return np.array(ups), state

def report(tag, ups, state):
    d = state.data
    up = ups[-1]
    h = np.array(d.xpos[:, TORSO, 2]) - consts.FOOT_RADIUS
    feet = np.array(d.sensordata[:, env._feet_contact_adr] > 0)
    trunk = np.array(d.sensordata[:, env._trunk_contact_adr] > 0)
    q = np.array(d.qpos[:, 7:]) - consts.DEFAULT_POSE
    dev = (q.reshape(B, 4, 3) ** 2).sum(-1)
    print(f"=== {tag} ===")
    print("  水平度: " + " ".join(
        f"{t:.1f}s {ups[min(int(t/env.dt), len(ups)-1)].mean():+.2f}"
        for t in (0.5, 1, 2, 4, 6, 8) if t < T * env.dt + 1e-9))
    print(f"  最終  正立(>0.85) {100*np.mean(up > 0.85):5.1f}%  "
          f"水平度 中央 {np.median(up):+.2f}  胴体高 中央 {1000*np.median(h):.0f}mm")
    print(f"        接地脚 {feet.sum(1).mean():.2f}/4  脚ごとの接地率 "
          f"[{' '.join(consts.LEGS)}] = {np.round(feet.mean(0), 2)}")
    print(f"        胴体が床に接触 {100*trunk.mean():.0f}%   "
          f"基準姿勢からの逸脱 = {np.round(dev.mean(0), 3)}")
    return h, feet, trunk

ups, state = run(False)
report("素の方策", ups, state)
qpos = np.array(state.data.qpos); qvel = np.array(state.data.qvel)

# 最終姿勢を 3 本ぶん、接触 geom まで含めて書き出す
d = mujoco.MjData(mm)
for i in range(min(3, B)):
    d.qpos[:] = qpos[i]; d.qvel[:] = qvel[i]; mujoco.mj_forward(mm, d)
    R = d.xmat[TORSO].reshape(3, 3)
    # 機体 x 軸 (鼻先) のワールドでの仰角。正なら鼻が上を向いている。
    nose = np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    print(f"\n  --- ep{i}  水平度 {R[2,2]:+.2f}  鼻先の仰角 {nose:+.0f}deg  "
          f"roll {roll:+.0f}deg  胴体高 {1000*(d.xpos[TORSO,2]-consts.FOOT_RADIUS):.0f}mm")
    for j, leg in enumerate(consts.LEGS):
        fz = d.site_xpos[env._feet_site_id[j], 2] * 1000 - consts.FOOT_RADIUS * 1000
        print(f"      {leg}: 基準差 {np.round(qpos[i,7+3*j:10+3*j]-consts.DEFAULT_POSE[3*j:3*j+3], 2)}"
              f"  足先高 {fz:6.1f}mm")
    names = sorted({
        f"{mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, d.contact.geom1[c])}"
        f"/{mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, d.contact.geom2[c])}"
        for c in range(d.ncon)})
    print(f"      床に触れている geom: {', '.join(names) if names else 'なし'}")

if not args.no_mirror:
    print()
    ups_m, state_m = run(True)
    report("左右反転した方策 (同じ初期状態)", ups_m, state_m)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n = min(3, B)
fig, axs = plt.subplots(2, n, figsize=(4 * n, 6), squeeze=False)
with mujoco.Renderer(mm, height=420, width=560) as r:
    for i in range(n):
        d.qpos[:] = qpos[i]; d.qvel[:] = qvel[i]; mujoco.mj_forward(mm, d)
        for row, (az, el) in enumerate(((90, -10), (0, -25))):
            v = mujoco.MjvCamera(); v.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.lookat[:] = d.xpos[TORSO]; v.distance = 0.8
            v.azimuth, v.elevation = az, el
            r.update_scene(d, camera=v)
            axs[row][i].imshow(r.render()); axs[row][i].axis("off")
            axs[row][i].set_title(f"ep{i} up={ups[-1,i]:+.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "stuck.png"), dpi=100)
print(f"\n-> {OUT}/stuck.png")
