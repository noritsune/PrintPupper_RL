"""指定ポリシーの遊脚最高到達点を実測する (地形上・地面相対)。"""
import argparse, os, pickle, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--step-height", type=float, default=0.03)
ap.add_argument("--difficulty", type=float, default=1.0)
ap.add_argument("--envs", type=int, default=128)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--vx", type=float, default=0.15)
args = ap.parse_args()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")
from env import _compat; _compat.install()
import jax, jax.numpy as jp, numpy as np
from env.terrain import PupperTerrain, terrain_config
from env import constants as consts
from scripts.policy import make_network
from brax.training.agents.ppo import networks as ppo_networks

N = args.envs
cfg = terrain_config(step_height=args.step_height)
cfg.terrain.difficulty = args.difficulty
cfg.naconmax = 8 * N
env = PupperTerrain(config=cfg)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
inf = jax.jit(ppo_networks.make_inference_fn(make_network(env))(
    pickle.load(open(args.ckpt, "rb")), deterministic=True))
st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), N))
step = jax.jit(jax.vmap(env.step))
peaks, alive_frac = [], []
for k in range(args.steps):
    st.info["command"] = jp.tile(jp.array([args.vx, 0., 0.]), (N, 1))
    a, _ = inf(st.obs, jax.random.PRNGKey(k))
    prev_peak = np.array(st.info["swing_peak"])
    st = step(st, a)
    if k < 100:
        continue
    # 接地の瞬間 (first_contact) の直前の swing_peak が、その遊脚の最高到達点
    contact = np.array(st.data.sensordata[:, env._feet_contact_adr] > 0)
    td = contact & (prev_peak > consts.FOOT_RADIUS + 1e-4)
    ok = np.array(st.done) == 0
    peaks.append(prev_peak[td & ok[:, None]] - consts.FOOT_RADIUS)
    alive_frac.append(ok.mean())
p = np.concatenate(peaks) * 1000
print(f"\nckpt: {args.ckpt}")
print(f"地形: 段差 {args.step_height*1000:.0f}mm difficulty {args.difficulty}  "
      f"前進 {args.vx} m/s  生存率 {100*np.mean(alive_frac):.0f}%")
print(f"足上げ目標 (max_foot_height) {cfg.reward_config.max_foot_height*1000:.0f} mm")
print(f"遊脚最高到達点 {len(p)} サンプル:")
print(f"   平均 {p.mean():5.1f} mm   中央 {np.median(p):5.1f}   "
      f"25% {np.percentile(p,25):5.1f}   75% {np.percentile(p,75):5.1f}   "
      f"90% {np.percentile(p,90):5.1f}   最大 {p.max():5.1f}")
for h in (10, 20, 30, 40, 50):
    print(f"   {h}mm 以上あがった遊脚の割合: {100*(p>=h).mean():5.1f} %")
