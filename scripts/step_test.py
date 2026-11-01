"""単一段差を何 mm まで越えられるかを直接測る。

地形学習の良し悪しは reward では読めない (平地を歩くだけでも報酬は稼げる)。
ここでは「高さ h の段を正面から越えられたか」だけを見る。

やり方: hfield を y 方向の帯に分け、帯ごとに違う段差高さを置く。段は x=0 に
1段だけ。機体を x=-0.8 の平地側に置き、前進 0.2 m/s を指令して、段の向こう側
(x > SUCCESS_X) まで到達できたかを数える。帯で分けているので 1 モデル 1 バッチで
全高さを同時に測れる。
"""
import argparse, os, pickle, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--heights", default="0,10,20,30,40,50,60,70,80",
                help="試す段差高さ [mm] のカンマ区切り")
ap.add_argument("--trials", type=int, default=24, help="高さあたりの試行数")
ap.add_argument("--secs", type=float, default=10.0)
ap.add_argument("--vx", type=float, default=0.2, help="前進指令 [m/s]")
ap.add_argument("--band", type=float, default=1.6, help="帯の幅 [m]")
ap.add_argument("--video", default=None, help="この高さ [mm] の 1 試行を動画に")
ap.add_argument("--gait-freq", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="歩容周期の範囲 [Hz]。**測る方策の学習時と同じ範囲にすること**。"
                     "reset が この範囲から一様に引くので、違う範囲で測ると方策が"
                     "知らない周期を渡すことになる (e40/g30 は 0.7 1.2 で学習)。"
                     "既定は env の設定値 (terrain=1.0-2.0, unified=0.7-1.2)")
ap.add_argument("--obs-history", type=int, default=-1, metavar="L",
                help="観測履歴スタックのフレーム数。既定の -1 は **ckpt の観測次元から"
                     "自動判定** する (38 + L*21)。手で渡す必要はほぼ無い")
ap.add_argument("--obs-stride", type=int, default=4, metavar="S",
                help="観測履歴の間引き間隔。次元からは復元できないので、学習時に "
                     "--obs-stride を既定 (4) 以外にした run ではここで合わせること")
ap.add_argument("--env", default="terrain", choices=["terrain", "unified"],
                help="terrain=段差専用の方策 (env/terrain.py), "
                     "unified=段差歩行+転倒復帰の両立方策 (env/unified.py)。"
                     "可動域の決め方と脚の衝突形状が違うので、方策に合わせること")
args = ap.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".4")

from env import _compat; _compat.install()
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from env import constants as consts
from env.terrain import PupperTerrain, set_hfield, terrain_config
from env.unified import PupperUnified, unified_config
from scripts.policy import make_network
from brax.training.agents.ppo import networks as ppo_networks

HEIGHTS = np.array([float(h) / 1000.0 for h in args.heights.split(",")])
SPAWN_X = -0.8
SUCCESS_X = 0.35          # 段の向こう側。機体長 0.29m ぶん完全に乗り越えた位置


# 両立版の方策は脚の衝突形状を持つ XML と傾き連動の可動域で学習しているので、
# ベンチもその環境で回す (env/terrain.py の環境に載せると可動域が 0.25 固定に
# なり、別の方策を測ることになる)。
BASE = PupperUnified if args.env == "unified" else PupperTerrain


class StepBench(BASE):
    """x=0 に 1 段だけ段差。高さは y の帯ごとに変える。"""

    def _edit_model(self, mj_model: mujoco.MjModel) -> None:
        n, half = consts.TERRAIN_N, consts.TERRAIN_HALF
        cell = 2 * half / (n - 1)
        hm = np.zeros((n, n), np.float32)          # [row=y, col=x]
        x = -half + np.arange(n) * cell
        y = -half + np.arange(n) * cell
        for i, h in enumerate(HEIGHTS):
            y0 = self.band_y(i) - args.band / 2
            rows = (y >= y0) & (y < y0 + args.band)
            hm[np.ix_(rows, x >= 0.0)] = h
        set_hfield(mj_model, hm)
        self._t_map = jp.asarray(hm)
        self._t_n = n
        self._t_cell = cell
        self._t_max = float(hm.max())

    def _get_termination(self, data, info):
        """地形外への打ち切りを外す。

        両立版 (--env unified) の環境自体は転倒では終了しないが、**このベンチ
        では転倒判定を残す** (super(PupperTerrain, self) は MRO 上
        PupperJoystick を指すので、姿勢と胴体接触による終了がそのまま効く)。
        段差専用の方策と同じ土俵で「越えられる段差」を測るためで、
        ここを外すと「一度転んでから起き上がって越えた」ものまで成功に
        数えてしまい、過去の run と比較できなくなる。
        復帰込みの成績は scripts/unified_demo.py で別に測る。

        PupperTerrain は |x| or |y| > TERRAIN_HALF-1.0 (=7m) で打ち切るが、
        このベンチは帯を y 方向に並べるので、端の帯 (0mm は中心 y=-6.4m、
        帯は -7.2〜-5.6m) がその打ち切り線と重なる。横にドリフトしただけの
        個体が「転倒」に数えられ、周期の遅い方策ほど不利に出てしまう。
        ここは自前の地形なので、転倒判定 (姿勢と胴体接触) だけを見る。
        """
        return super(PupperTerrain, self)._get_termination(data, info)

    @staticmethod
    def band_y(i):
        """帯 i の中心 y。地形中央 (-half+..) に等間隔で並べる。"""
        return (i - (len(HEIGHTS) - 1) / 2) * args.band


def main():
    if args.env == "unified":
        cfg = unified_config()
        cfg.recover.p_walk = 1.0          # 立った状態から始める (歩行だけを測る)
        per_env = 16
    else:
        cfg = terrain_config()
        per_env = 8
    if args.gait_freq is not None:
        cfg.gait_freq = list(args.gait_freq)
    # 観測履歴スタック。学習時と違う値で env を組むと次元不一致で落ちる。
    from policy import ckpt_obs_history
    hist = ckpt_obs_history(args.ckpt) if args.obs_history < 0 else args.obs_history
    cfg.obs_history.length = hist
    cfg.obs_history.stride = args.obs_stride
    if hist:
        print(f"[obs] 履歴 length={hist} stride={args.obs_stride} "
              f"({hist*args.obs_stride*cfg.ctrl_dt:.2f}s) -> 観測 {38+hist*21} 次元")
    cfg.terrain.difficulty = 0.0          # 段は _edit_model で直接書くので地形生成は切る
    n_env = len(HEIGHTS) * args.trials
    cfg.naconmax = per_env * n_env
    env = StepBench(config=cfg)
    env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})

    net = make_network(env)
    params = pickle.load(open(args.ckpt, "rb"))
    inference = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))

    # --- 初期状態: 全試行を平地側に、段に正対させて置く ---
    keys = jax.random.split(jax.random.PRNGKey(0), n_env)
    state = jax.jit(jax.vmap(env.reset))(keys)

    idx = np.repeat(np.arange(len(HEIGHTS)), args.trials)
    ys = np.array([StepBench.band_y(i) for i in idx])
    rng = np.random.default_rng(0)
    ys = ys + rng.uniform(-0.15, 0.15, n_env)        # 帯の中で横位置をばらす
    xs = SPAWN_X + rng.uniform(-0.05, 0.05, n_env)

    qpos = state.data.qpos
    qpos = qpos.at[:, 0].set(jp.asarray(xs))
    qpos = qpos.at[:, 1].set(jp.asarray(ys))
    qpos = qpos.at[:, 2].set(consts.STAND_HEIGHT + consts.FOOT_RADIUS)
    qpos = qpos.at[:, 3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))     # yaw = 0 (+x を向く)
    qpos = qpos.at[:, 7:].set(jp.asarray(env._default_pose))
    data = state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
    data = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))(env.mjx_model, data)
    state = state.replace(data=data)
    state.info["command"] = jp.tile(jp.array([args.vx, 0.0, 0.0]), (n_env, 1))

    step = jax.jit(jax.vmap(env.step))
    n = int(args.secs / env.dt)
    alive = np.ones(n_env, bool)
    max_x = np.full(n_env, SPAWN_X)
    frames_env = None
    if args.video is not None:
        want = float(args.video) / 1000.0
        cand = np.where((np.abs(HEIGHTS[idx] - want) < 1e-9))[0]
        frames_env = int(cand[0]) if len(cand) else None
    traj = []

    for k in range(n):
        state.info["command"] = jp.tile(jp.array([args.vx, 0.0, 0.0]), (n_env, 1))
        act, _ = inference(state.obs, jax.random.PRNGKey(k))
        state = step(state, act)
        alive &= np.array(state.done) == 0.0
        x = np.array(state.data.qpos[:, 0])
        max_x = np.maximum(max_x, np.where(alive, x, -np.inf))
        if frames_env is not None:
            traj.append(jax.tree_util.tree_map(lambda v: v[frames_env], state))

    print(f"\nckpt: {args.ckpt}")
    print(f"前進指令 {args.vx} m/s, {args.secs:.0f}s, 高さあたり {args.trials} 試行")
    print(f"成功判定: 段 (x=0) を越えて x > {SUCCESS_X} m に到達し、転倒しないこと\n")
    print("  段差    成功率   到達 x 中央値   生存率")
    print("  " + "-" * 44)
    for i, h in enumerate(HEIGHTS):
        m = idx == i
        ok = (max_x[m] > SUCCESS_X) & alive[m]
        print(f"  {h*1000:3.0f}mm   {100*ok.mean():5.1f} %   "
              f"{np.median(max_x[m]):+7.2f} m      {100*alive[m].mean():5.1f} %")
    # 成功率 50% を超える最大の高さ
    rates = np.array([((max_x[idx == i] > SUCCESS_X) & alive[idx == i]).mean()
                      for i in range(len(HEIGHTS))])
    good = HEIGHTS[rates >= 0.5]
    print(f"\n  成功率 50% 以上の最大段差: "
          f"{good.max()*1000:.0f} mm" if len(good) else "\n  50% を超えた高さなし")

    if frames_env is not None and traj:
        import imageio
        frames = env.render(traj[::2], height=360, width=640, camera="track")
        out = os.path.join(ROOT, "out", f"step_test_{args.video}mm.mp4")
        imageio.mimsave(out, frames, fps=int(1 / env.dt / 2), quality=8)
        print(f"  動画 -> {out}")


if __name__ == "__main__":
    main()
