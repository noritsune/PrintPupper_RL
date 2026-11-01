"""対角2脚での静止保持 (env/balance.py) の評価。

測るもの:
  1. 保持時間     何秒立っていられるか。分布と、上限 (エピソード長) 到達率
  2. 失敗の仕方   遊脚の接地で終わったのか、大きく傾いて終わったのか
  3. 外乱耐性     どれくらいの速度キックまで耐えるか
  4. 実機成立性   サーボのトルク飽和 / PWM 飽和 / 4節リンクの実現可能性

使い方:
  python scripts/balance_test.py --run bal1
  python scripts/balance_test.py --ckpt out/train/bal1/ckpt/latest.pkl --episodes 256
"""
import argparse, functools, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--run", default=None, help="out/train/<run>")
ap.add_argument("--episodes", type=int, default=128, help="同時に回すエピソード数")
ap.add_argument("--no-randomize", action="store_true",
                help="ドメインランダム化を切る (公称モデルだけで測る)")
ap.add_argument("--kicks", default="0.1,0.2,0.3,0.4,0.6",
                help="外乱テストの速度キック [m/s]")
ap.add_argument("--cap", type=float, default=None,
                help="1エピソードの上限 [s] (既定は学習時と同じ 20s)。"
                     "上限に張り付いたら伸ばして本当の保持時間を測る")
ap.add_argument("--cycles", type=float, default=3.0,
                help="上限の何本ぶん回すか。上限を伸ばしたときは 1-2 で足りる")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import jax, jax.numpy as jp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from env import _compat; _compat.install()
import warp as wp; wp.config.quiet = True
from mujoco_playground._src import wrapper

from env import constants as consts
from env.balance import PupperBalance, balance_config
from env.deploy import joint_to_servo, pwm_saturated
from env.linkage import infeasibility
from env.randomize import domain_randomize
from policy import load

if args.run is None and args.ckpt is None:
    args.run = sorted(os.listdir(os.path.join(ROOT, "out", "train")))[-1]
ckpt = args.ckpt or os.path.join(ROOT, "out", "train", args.run, "ckpt", "latest.pkl")
OUT = args.out or os.path.join(ROOT, "out", "eval", (args.run or "ckpt") + "_balance")
os.makedirs(OUT, exist_ok=True)

B = args.episodes
cfg = balance_config()
if args.cap is not None:
    cfg.episode_length = int(round(args.cap / cfg.ctrl_dt))
cfg.naconmax = 8 * B
env = PupperBalance(config=cfg)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
_, inference, _ = load(ckpt, env=env)
DT = env.dt
CAP = cfg.episode_length * DT
print(f"[eval] {ckpt}\n[out ] {OUT}\n[env ] 保持の上限 {CAP:.0f}s  "
      f"{B} エピソード同時  ランダム化 {'なし' if args.no_randomize else 'あり'}\n")
report = {"ckpt": ckpt, "cap_s": CAP, "episodes": B}

# センサのアドレスを控える (vmap の中から mj_model を引かないため)
mm = env.mj_model
env._up_adr = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id]


# ================================================================ 1. 保持時間
# 学習と同じラッパ (ドメインランダム化 + 自動リセット) で回し、done が立つ
# たびにエピソード長を拾う。full_reset=True にして、毎回きちんと初期状態を
# 引き直す (既定の自動リセットは「最初の状態」を使い回すので、同じ初期姿勢
# ばかりを測ってしまう)。
# ppo.train は randomization_fn に rng を詰めてから渡す。ここでは自前で詰める。
rand_fn = None if args.no_randomize else functools.partial(
    domain_randomize, rng=jax.random.split(jax.random.PRNGKey(42), B))
wrapped = wrapper.wrap_for_brax_training(
    env, episode_length=cfg.episode_length, action_repeat=1,
    randomization_fn=rand_fn, full_reset=True)
w_reset, w_step = jax.jit(wrapped.reset), jax.jit(wrapped.step)

T = int(round(CAP / DT * args.cycles))     # 上限に達しなければこの間に何本も入る
state = w_reset(jax.random.split(jax.random.PRNGKey(0), B))
holds, trunc, ends = [], [], []
print(f"1. 保持時間  ({T} step = {T*DT:.0f}s ぶん回す)")
# 注意: done が立ったステップの state.data / info は、full_reset の自動リセットに
# よって既に「次のエピソードの初期状態」に差し替わっている (steps だけは
# ラッパが明示的に引き継ぐ)。終わり方の判定には直前ステップの姿勢を使う。
prev_up = np.ones(B)
for k in range(T):
    act, _ = inference(state.obs, jax.random.PRNGKey(k))
    state = w_step(state, act)
    done = np.array(state.done) > 0
    if done.any():
        steps = np.array(state.info["steps"])[done]
        tr = steps >= cfg.episode_length      # 上限による打ち切り
        holds += list(steps * DT); trunc += list(tr)
        # 失敗の型: 直前に大きく傾いていれば転倒、そうでなければ遊脚の接地
        ends += list(np.where(tr, 2, np.where(prev_up[done] < 0.5, 1, 0)))
    prev_up = np.array(state.data.sensordata[:, env._up_adr + 2])
holds = np.array(holds); trunc = np.array(trunc); ends = np.array(ends)
if len(holds) == 0:
    sys.exit("エピソードが1つも終わらなかった (上限到達を含め異常)")
q = np.percentile(holds, [10, 50, 90])
print(f"  エピソード {len(holds)} 本")
print(f"  保持 平均 {holds.mean():5.2f}s  中央 {q[1]:5.2f}s  "
      f"10%点 {q[0]:5.2f}s  90%点 {q[2]:5.2f}s  最長 {holds.max():5.2f}s")
print(f"  上限 {CAP:.0f}s 到達 {100*trunc.mean():5.1f}%  "
      f"| 終わり方: 遊脚接地 {100*np.mean(ends==0):4.1f}%  "
      f"転倒 {100*np.mean(ends==1):4.1f}%  上限 {100*np.mean(ends==2):4.1f}%")
report["hold"] = dict(n=len(holds), mean=float(holds.mean()),
                      median=float(q[1]), p10=float(q[0]), p90=float(q[2]),
                      max=float(holds.max()), cap_reached=float(trunc.mean()),
                      end_swing_touch=float(np.mean(ends == 0)),
                      end_fall=float(np.mean(ends == 1)))

# ================================================================ 2. 外乱耐性
# 自動リセットの無い素の env で、2s 立たせてから水平方向に速度キックを入れ、
# さらに 2s 立っていられるかを見る。
print("\n2. 外乱耐性 (2s 保持 -> キック -> さらに 2s)")
v_reset = jax.jit(jax.vmap(env.reset))
v_step = jax.jit(jax.vmap(env.step))
PRE, POST = int(2.0 / DT), int(2.0 / DT)
kick_res = {}
for mag in [float(x) for x in args.kicks.split(",")]:
    st = v_reset(jax.random.split(jax.random.PRNGKey(1), B))
    alive = np.ones(B, bool)
    for k in range(PRE):
        a, _ = inference(st.obs, jax.random.PRNGKey(k))
        st = v_step(st, a)
        alive &= np.array(st.done) == 0
    ang = np.linspace(0, 2 * np.pi, B, endpoint=False)   # 全方位に散らす
    dv = jp.array(np.stack([np.cos(ang), np.sin(ang), np.zeros(B)], 1) * mag)
    st = st.replace(data=st.data.replace(qvel=st.data.qvel.at[:, :3].add(dv)))
    survived = alive.copy()
    for k in range(POST):
        a, _ = inference(st.obs, jax.random.PRNGKey(PRE + k))
        st = v_step(st, a)
        survived &= np.array(st.done) == 0
    base = max(alive.sum(), 1)
    kick_res[mag] = float(survived.sum() / base)
    print(f"  キック {mag:.2f} m/s  キック前に立てていた {alive.sum():3d}/{B} 本のうち "
          f"{survived.sum():3d} 本が耐えた ({100*kick_res[mag]:5.1f}%)")
report["kick_survival"] = kick_res

# ================================================================ 3. 実機成立性
print("\n3. 実機成立性 (公称モデルで 1 エピソード)")
s_reset, s_step = jax.jit(env.reset), jax.jit(env.step)
st = s_reset(jax.random.PRNGKey(0))
q_ref = np.array(st.info["q_ref"])
rec = {k: [] for k in ("qpos", "qvel", "tau", "act", "grav", "footz", "contact")}
hold_single = 0.0
for k in range(int(CAP / DT)):
    a, _ = inference(st.obs, jax.random.PRNGKey(k))
    st = s_step(st, a)
    d = st.data
    rec["qpos"].append(np.array(d.qpos)); rec["qvel"].append(np.array(d.qvel))
    rec["tau"].append(np.array(d.actuator_force)); rec["act"].append(np.array(a))
    rec["grav"].append(np.array(env._gravity(d)))
    rec["footz"].append(np.array(d.site_xpos[env._feet_site_id][:, 2]))
    rec["contact"].append(np.array(d.sensordata[env._feet_contact_adr] > 0))
    if float(st.done) > 0:
        hold_single = (k + 1) * DT
        break
else:
    hold_single = CAP
rec = {k: np.array(v) for k, v in rec.items()}
targets = q_ref + rec["act"] * cfg.action_scale        # 実機に出す関節指令
servo = np.array([joint_to_servo(t) for t in targets])
pwm_sat = np.array([pwm_saturated(s) for s in servo])
inf = np.array([float(infeasibility(jp.array(q[7:]))) for q in rec["qpos"]])
tau = np.abs(rec["tau"])
stance = np.array(st.info["stance"]) > 0.5
print(f"  この1本の保持 {hold_single:.2f}s")
print(f"  トルク 最大 {tau.max():.2f} Nm  平均 {tau.mean():.2f} Nm  "
      f"飽和 (>2.985Nm) {100*np.mean(tau > 2.985):.1f}%  "
      f"軟限界 (>1.5Nm) 超え {100*np.mean(tau > 1.5):.1f}%")
print(f"  関節速度 最大 {np.abs(rec['qvel'][:, 6:]).max():.1f} rad/s "
      f"(DS3235 無負荷 6.0-8.2)")
print(f"  PWM 飽和 {100*pwm_sat.mean():.2f}%  4節リンク実現不能 "
      f"{100*np.mean(inf > 1e-6):.2f}% (最大 {inf.max():.3f} rad)")
print(f"  接地脚の接地率 {100*rec['contact'][:, stance].mean():.1f}%  "
      f"遊脚クリアランス 平均 {1000*(rec['footz'][:, ~stance] - consts.FOOT_RADIUS).mean():.0f}mm")
report["hardware"] = dict(
    hold_s=hold_single, tau_max=float(tau.max()), tau_mean=float(tau.mean()),
    tau_sat_frac=float(np.mean(tau > 2.985)),
    tau_over_soft_frac=float(np.mean(tau > 1.5)),
    qvel_max=float(np.abs(rec["qvel"][:, 6:]).max()),
    pwm_sat_frac=float(pwm_sat.mean()),
    linkage_infeasible_frac=float(np.mean(inf > 1e-6)),
    stance_contact_frac=float(rec["contact"][:, stance].mean()))

# ================================================================ 出力
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(holds, bins=np.linspace(0, CAP, 41), color="tab:blue")
ax[0].axvline(holds.mean(), color="tab:red", lw=2,
              label=f"mean {holds.mean():.2f}s")
ax[0].set_title(f"hold time [s]  (n={len(holds)})"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("s")
ks = sorted(kick_res); ax[1].plot(ks, [100 * kick_res[k] for k in ks], "o-", lw=2)
ax[1].set_title("survival after a push [%]"); ax[1].set_xlabel("kick [m/s]")
ax[1].set_ylim(-5, 105)
t = np.arange(len(rec["grav"])) * DT
ax[2].plot(t, np.linalg.norm(rec["grav"][:, :2], axis=1), lw=1.5, label="tilt |g_xy|")
for i, lbl in enumerate(consts.LEGS):
    if not stance[i]:
        ax[2].plot(t, (rec["footz"][:, i] - consts.FOOT_RADIUS) * 10, lw=1,
                   label=f"{lbl} clearance x10 [m]")
ax[2].set_title("one episode"); ax[2].set_xlabel("s"); ax[2].legend(fontsize=7)
for a_ in ax: a_.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "balance.png"), dpi=110)
plt.close(fig)

if not args.no_video:
    import imageio
    mj = env.mj_model
    dd = mujoco.MjData(mj)
    frames = []
    with mujoco.Renderer(mj, height=480, width=640) as r:
        for i in range(0, len(rec["qpos"]), 2):
            dd.qpos[:] = rec["qpos"][i]; dd.qvel[:] = rec["qvel"][i]
            mujoco.mj_forward(mj, dd)
            r.update_scene(dd, camera="track")
            frames.append(r.render())
    if frames:
        imageio.mimsave(os.path.join(OUT, "balance.mp4"), frames,
                        fps=int(1 / DT / 2), quality=8)

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(f"\n-> {OUT}/balance.png, report.json"
      f"{'' if args.no_video else ', balance.mp4'}")
