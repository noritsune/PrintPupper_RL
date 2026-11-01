"""方策が「詰まる」状態を記録して、次の run の初期姿勢に混ぜられる形で保存する。

使い方:
  python scripts/make_stuck_states.py --ckpt out/train/rec4/ckpt/final.pkl \
      --out out/stuck_rec4.npz
  python scripts/train.py --env recover --tag rec8 \
      --stuck-states out/stuck_rec4.npz --rec p_stuck=0.25 ...

なぜ要るか:
  「その姿勢でしか練習できない動作」は、そこへ至る行動と絡んで信用割当が
  難しいうえ、1エピソードあたりの試行数が少ない。rec4 では「伏せ」を初期
  姿勢に直接入れることで 伏せ->立つ が 78.6% -> 86.0% になった。ここでは
  パラメータで作った近似ではなく、**前の run が実際に止まった状態そのもの**
  (qpos / qvel / そのときの行動) を記録して使う。逆立ちのように「方策が
  能動的に保っている」姿勢は、姿勢と速度を指定しただけでは再現できない。

記録する条件: エピソード終盤で「まだ復帰していない」(水平度 < upright_cos)
状態。--every でエピソードの後半から間引いて集める。
"""
import argparse, functools, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "out/train/rec4/ckpt/final.pkl"))
ap.add_argument("--out", default=os.path.join(ROOT, "out", "stuck.npz"))
ap.add_argument("--episodes", type=int, default=256)
ap.add_argument("--from-secs", type=float, default=2.0,
                help="この時刻以降の状態だけ集める (転がっている途中は要らない)")
ap.add_argument("--every", type=int, default=25, help="何 step ごとに 1 枚取るか")
ap.add_argument("--max", type=int, default=2048, help="保存する状態の上限")
ap.add_argument("--contacts-per-env", type=int, default=16)
ap.add_argument("--env", default="recover", choices=["recover", "unified"],
                help="recover=復帰専用 (env/recover.py), "
                     "unified=段差歩行+転倒復帰の両立 (env/unified.py)")
ap.add_argument("--difficulty", type=float, default=0.0,
                help="--env unified のとき。0=平地, 1.0=学習と同じ段差地形")
args = ap.parse_args()

import jax, jax.numpy as jp
from env import _compat; _compat.install()
import warp as wp
from mujoco_playground._src import wrapper
from env import constants as consts
from env.randomize import domain_randomize
from env.recover import PupperRecover, recover_config
from env.unified import PupperUnified, unified_config
from policy import load

B = args.episodes
UNIFIED = args.env == "unified"
cfg = unified_config() if UNIFIED else recover_config()
if UNIFIED:
    cfg.recover.p_walk = 0.0            # 必ず転倒から始める
    cfg.terrain.difficulty = args.difficulty
    # 両立版は 1 エピソード 20s。詰まり状態は数秒で十分なので短くする
    # (長く回すと「詰まった」ではなく「立って歩いている」状態が混ざる)。
    cfg.episode_length = 400
# 詰まるのは転がってから。仰向けと横倒しだけを入口にする。
cfg.recover.p_flip, cfg.recover.p_side, cfg.recover.p_prone = 0.8, 0.2, 0.0
cfg.recover.p_stuck = 0.0
cfg.naconmax = args.contacts_per_env * B
env = (PupperUnified if UNIFIED else PupperRecover)(config=cfg)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
_, inference, _ = load(args.ckpt, env=env)
T = cfg.episode_length
mm = env.mj_model
UP = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id] + 2

rand_fn = functools.partial(domain_randomize,
                            rng=jax.random.split(jax.random.PRNGKey(1), B))
wrapped = wrapper.wrap_for_brax_training(
    env, episode_length=T, action_repeat=1, randomization_fn=rand_fn, full_reset=True)
w_reset, w_step = jax.jit(wrapped.reset), jax.jit(wrapped.step)

print(f"[stuck] {args.ckpt}  {B} エピソード x {T*env.dt:.0f}s")
state = w_reset(jax.random.split(jax.random.PRNGKey(0), B))
k0 = int(args.from_secs / env.dt)
qp, qv, ac = [], [], []
for k in range(T - 1):
    act, _ = inference(state.obs, jax.random.PRNGKey(k))
    state = w_step(state, act)
    if k >= k0 and (k - k0) % args.every == 0:
        up = np.array(state.data.sensordata[:, UP])
        sel = up < cfg.recover.upright_cos          # まだ復帰していない個体だけ
        if sel.any():
            qp.append(np.array(state.data.qpos)[sel])
            qv.append(np.array(state.data.qvel)[sel])
            ac.append(np.array(act)[sel])

qp = np.concatenate(qp); qv = np.concatenate(qv); ac = np.concatenate(ac)
if len(qp) > args.max:
    idx = np.random.default_rng(0).choice(len(qp), args.max, replace=False)
    qp, qv, ac = qp[idx], qv[idx], ac[idx]
# 位置は原点に寄せる (地形は平地なので x,y は意味を持たない)
qp[:, 0:2] = 0.0
np.savez_compressed(args.out, qpos=qp, qvel=qv, act=ac)

import mujoco
d = mujoco.MjData(mm)
ups, hs, feet, trunk = [], [], [], []
for i in range(0, len(qp), max(1, len(qp) // 200)):
    d.qpos[:] = qp[i]; d.qvel[:] = qv[i]; mujoco.mj_forward(mm, d)
    R = d.xmat[mm.body(consts.ROOT_BODY).id].reshape(3, 3)
    ups.append(R[2, 2])
    hs.append(d.xpos[mm.body(consts.ROOT_BODY).id, 2] - consts.FOOT_RADIUS)
    feet.append((d.sensordata[env._feet_contact_adr] > 0).sum())
    trunk.append(d.sensordata[env._trunk_contact_adr] > 0)
print(f"[stuck] {len(qp)} 状態 -> {args.out}")
print(f"        水平度 中央 {np.median(ups):+.2f}  胴体高 中央 "
      f"{1000*np.median(hs):.0f}mm  接地脚 中央 {np.median(feet):.0f}/4  "
      f"胴体接地 {100*np.mean(trunk):.0f}%")
