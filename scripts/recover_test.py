"""転倒復帰 (env/recover.py) の評価。

測るもの:
  1. 復帰成功率   初期姿勢 (仰向け / 横倒し / 正立) ごとの成功率
  2. 復帰時間     成功したエピソードが何秒で立ち上がったか
  3. 復帰後の維持 立ってから最後まで立っていられたか (再転倒していないか)
  4. 実機成立性   トルク飽和 / PWM 飽和 / 4節リンクの実現可能性 / 関節速度

「成功」の定義は学習時と同じ: 水平度 > 0.85 かつ 胴体高さ 誤差 30mm 以内 かつ
接地脚 3本以上 かつ ほぼ静止、を 0.5s 連続で満たすこと。

使い方:
  python scripts/recover_test.py --run rec1
  python scripts/recover_test.py --ckpt out/train/rec1/ckpt/final.pkl --episodes 256
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
ap.add_argument("--secs", type=float, default=None,
                help="1エピソードの長さ [s] (既定は学習時と同じ 8s)")
ap.add_argument("--flip-only", action="store_true",
                help="全エピソードを仰向けで始める (いちばん難しい条件だけ測る)")
ap.add_argument("--contacts-per-env", type=int, default=16)
ap.add_argument("--impl", default="warp", choices=["warp", "jax"],
                help="jax にすると CPU でも動く (学習中に動作確認したいとき)")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--env", default="recover", choices=["recover", "unified"],
                help="recover=復帰専用の方策 (env/recover.py), "
                     "unified=段差歩行+転倒復帰の両立方策 (env/unified.py)")
ap.add_argument("--difficulty", type=float, default=0.0,
                help="--env unified のとき。0=平地 (rec* と同条件で比較できる), "
                     "1.0=学習と同じ段差地形の上で転倒させる")
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
from env.deploy import joint_to_servo, pwm_saturated
from env.linkage import infeasibility
from env.randomize import domain_randomize
from env.recover import PupperRecover, recover_config
from env.unified import PupperUnified, unified_config
from policy import load

if args.run is None and args.ckpt is None:
    args.run = sorted(os.listdir(os.path.join(ROOT, "out", "train")))[-1]
ckpt = args.ckpt or os.path.join(ROOT, "out", "train", args.run, "ckpt", "final.pkl")
if not os.path.exists(ckpt):
    ckpt = ckpt.replace("final.pkl", "latest.pkl")
OUT = args.out or os.path.join(ROOT, "out", "eval", (args.run or "ckpt") + "_recover")
os.makedirs(OUT, exist_ok=True)

B = args.episodes
UNIFIED = args.env == "unified"
cfg = unified_config() if UNIFIED else recover_config()
if UNIFIED:
    # 初期姿勢は必ず転倒側にし、内訳を rec* の評価と同じ比率に合わせる
    # (両立版の p_flip/p_side/p_prone は「転倒側の中での」条件付き確率)。
    cfg.recover.p_walk = 0.0
    cfg.terrain.difficulty = args.difficulty
# **評価の初期姿勢の内訳はここで固定する**。以前は env/recover.py の現在値を
# そのまま使っていたので、run ごとに学習側の設定を振ると評価条件まで一緒に
# 動いてしまい、run 間の比較が成立していなかった (rec4 と rec6 の差が条件の
# 違いだった)。学習側に何を足してもこの行の値は変えないこと。
cfg.recover.p_flip, cfg.recover.p_side, cfg.recover.p_prone = 0.65, 0.15, 0.15
if UNIFIED:
    # 条件付き確率なので合計を 1 にする。0.05 ぶんは「正立」に当たるが、
    # 両立版ではそれは歩行スポーン (p_walk) の担当なので伏せに寄せる。
    cfg.recover.p_prone = 0.20
cfg.recover.p_stuck = 0.0
if args.secs is not None:
    cfg.episode_length = int(round(args.secs / cfg.ctrl_dt))
if args.flip_only:
    cfg.recover.p_flip, cfg.recover.p_side = 1.0, 0.0
cfg.naconmax = args.contacts_per_env * B
cfg.impl = args.impl
env = (PupperUnified if UNIFIED else PupperRecover)(config=cfg)
if args.impl == "warp":
    env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
_, inference, _ = load(ckpt, env=env)
DT = env.dt
T = cfg.episode_length
rc = cfg.recover
HOLD = int(rc.hold_steps)
print(f"[eval] {ckpt}\n[out ] {OUT}")
print(f"[env ] 1エピソード {T*DT:.0f}s ({T} step)  {B} エピソード同時  "
      f"ランダム化 {'なし' if args.no_randomize else 'あり'}  "
      f"初期姿勢 {'仰向けのみ' if args.flip_only else '仰向け/横倒し/正立'}\n")
report = {"ckpt": ckpt, "episodes": B, "secs": T * DT}

mm = env.mj_model
UP_ADR = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id] + 2
LIN_ADR = mm.sensor_adr[mm.sensor(consts.LOCAL_LINVEL_SENSOR).id]
GYR_ADR = mm.sensor_adr[mm.sensor(consts.GYRO_SENSOR).id]
TORSO = mm.body(consts.ROOT_BODY).id

# =============================================== 1-3. 復帰の成否・時間・維持
rand_fn = None if args.no_randomize else functools.partial(
    domain_randomize, rng=jax.random.split(jax.random.PRNGKey(42), B))
wrapped = wrapper.wrap_for_brax_training(
    env, episode_length=T, action_repeat=1, randomization_fn=rand_fn,
    full_reset=True)
w_reset, w_step = jax.jit(wrapped.reset), jax.jit(wrapped.step)

state = w_reset(jax.random.split(jax.random.PRNGKey(0), B))
up = np.zeros((T, B)); hgt = np.zeros((T, B)); feet = np.zeros((T, B))
spd = np.zeros((T, B)); asp = np.zeros((T, B)); trunk = np.zeros((T, B))
qpos = np.zeros((T, B, mm.nq), np.float32); acts = np.zeros((T, B, 12), np.float32)
taus = np.zeros((T, B, 12), np.float32); qvels = np.zeros((T, B, mm.nv), np.float32)
up0 = np.array(state.data.sensordata[:, UP_ADR])       # 初期姿勢の水平度
print(f"1-3. 復帰 ({T} step = {T*DT:.0f}s)")
for k in range(T):
    if UNIFIED:
        # **速度コマンドを 0 に固定する**。両立版の env は歩行用のコマンドを
        # ランダムに出すので、そのまま測ると「起き上がった直後に歩けと言われ、
        # 歩いたせいで静止条件を外す」個体が失敗に数えられ、復帰専用の run
        # (rec*) と比較にならない。ここでは rec* と同じ「起きてその場で
        # 止まれるか」を測る。歩行まで含めた通しは scripts/unified_demo.py。
        state.info["command"] = jp.zeros((B, 3))
    act, _ = inference(state.obs, jax.random.PRNGKey(k))
    state = w_step(state, act)
    d = state.data
    up[k] = np.array(d.sensordata[:, UP_ADR])
    hgt[k] = np.array(d.xpos[:, TORSO, 2]) - consts.FOOT_RADIUS
    if UNIFIED and args.difficulty > 0:
        # 段差地形の上では胴体高を **足元の地面から** 測らないと、段に乗って
        # いるだけで「高すぎ」と判定されてしまう (成功判定は ±30mm)。
        hgt[k] -= np.array(env._ground_height(d.xpos[:, TORSO, :2]))
    feet[k] = np.array(d.sensordata[:, env._feet_contact_adr] > 0).sum(1)
    trunk[k] = np.array(d.sensordata[:, env._trunk_contact_adr] > 0)
    spd[k] = np.linalg.norm(np.array(d.sensordata[:, LIN_ADR:LIN_ADR + 3]), axis=1)
    asp[k] = np.linalg.norm(np.array(d.sensordata[:, GYR_ADR:GYR_ADR + 3]), axis=1)
    qpos[k] = np.array(d.qpos); qvels[k] = np.array(d.qvel)
    acts[k] = np.array(act); taus[k] = np.array(d.actuator_force)

# 最後の 1 ステップは捨てる。エピソード長で打ち切られた瞬間、ラッパーの
# 自動リセットで data が **次のエピソードの初期状態** に差し替わっており、
# そこを混ぜると「最終姿勢」が方策に依らず初期分布そのものになる
# (実際、全 run で完全に同じ数字が出ていた)。「復帰後の維持」の判定にも
# 効いて、立ち続けていた個体まで失敗に数えてしまう。
# balance_test.py の「done が立ったステップの state.data は既に次の
# エピソードの初期状態」と同じ罠。
up, hgt, feet = up[:-1], hgt[:-1], feet[:-1]
spd, asp, trunk = spd[:-1], asp[:-1], trunk[:-1]
qpos, qvels, acts, taus = qpos[:-1], qvels[:-1], acts[:-1], taus[:-1]
T = T - 1

ok = ((up > rc.upright_cos)
      & (np.abs(hgt - consts.STAND_HEIGHT) < rc.height_tol)
      & (feet >= rc.feet_needed)
      & (spd < rc.settle_lin) & (asp < rc.settle_ang))      # (T, B)

# 0.5s (HOLD step) 連続で条件を満たした最初の時刻
run = np.zeros(B, int); t_rec = np.full(B, np.nan)
for k in range(T):
    run = np.where(ok[k], run + 1, 0)
    hit = (run == HOLD) & np.isnan(t_rec)
    t_rec[hit] = (k + 1 - HOLD) * DT
success = ~np.isnan(t_rec)
# 復帰後に最後まで立っていられたか (末尾 0.5s が正立)
held = success & (up[-HOLD:].min(0) > rc.upright_cos)

cat = np.where(up0 < -0.5, 0, np.where(up0 < 0.5, 1, 2))    # 仰向け/横倒し/正立
names = ["仰向け", "横倒し", "正立   "]
print(f"  全体            成功 {100*success.mean():5.1f}%  "
      f"復帰後も維持 {100*held.mean():5.1f}%  "
      f"復帰時間 中央 {np.nanmedian(t_rec):5.2f}s")
by_cat = {}
for c in (0, 1, 2):
    s = cat == c
    if s.sum() == 0:
        continue
    tm = t_rec[s & success]
    print(f"  {names[c]} ({s.sum():3d}本)  成功 {100*success[s].mean():5.1f}%  "
          f"維持 {100*held[s].mean():5.1f}%  "
          f"復帰時間 中央 {np.median(tm) if len(tm) else float('nan'):5.2f}s  "
          f"90%点 {np.percentile(tm, 90) if len(tm) else float('nan'):5.2f}s")
    by_cat[names[c].strip()] = dict(
        n=int(s.sum()), success=float(success[s].mean()), held=float(held[s].mean()),
        t_median=float(np.median(tm)) if len(tm) else None,
        t_p90=float(np.percentile(tm, 90)) if len(tm) else None)
report["by_initial_pose"] = by_cat
report["overall"] = dict(success=float(success.mean()), held=float(held.mean()),
                         t_median=float(np.nanmedian(t_rec)))
print(f"  最終姿勢: 正立 {100*np.mean(up[-1] > rc.upright_cos):5.1f}%  "
      f"横倒し {100*np.mean(np.abs(up[-1]) <= rc.upright_cos):5.1f}%  "
      f"仰向けのまま {100*np.mean(up[-1] < -0.5):5.1f}%")
# 胴体が床に着いたまま止まっていないか。rec1-rec6 が仰向けから抜けられ
# なかったのは「胴体の前端を床に着けた逆立ち」(水平度 +0.54 / 前脚2本だけ
# 接地 / 胴体接地 100%) で止まっていたため。水平度だけ見ていると
# 「半分起きている」に見えるので、必ず並べて出す。
print(f"  最後の1秒: 胴体が床に接触 {100*trunk[-50:].mean():5.1f}% の時間  "
      f"接地脚 中央 {np.median(feet[-50:]):.1f}/4  "
      f"胴体高 中央 {1000*np.median(hgt[-50:]):5.0f}mm")
report["final"] = dict(
    upright=float(np.mean(up[-1] > rc.upright_cos)),
    trunk_contact=float(trunk[-50:].mean()),
    feet=float(np.median(feet[-50:])))

# =============================================== 4. 実機成立性
print("\n4. 実機成立性 (全エピソード・全ステップ)")
targets = consts.DEFAULT_POSE + acts * cfg.action_scale        # 実機に出す指令
flat = targets.reshape(-1, 12)
sub = flat[:: max(1, len(flat) // 20000)]                      # 間引いて評価
servo = np.array([joint_to_servo(t) for t in sub])
pwm_sat = np.array([pwm_saturated(s) for s in servo])
qsub = jp.array(qpos[:, :, 7:].reshape(-1, 12)[:: max(1, len(flat) // 20000)])
inf = np.array(jax.vmap(infeasibility)(qsub))
tau = np.abs(taus)
jv = np.abs(qvels[:, :, 6:])
print(f"  トルク 最大 {tau.max():.2f} Nm  平均 {tau.mean():.2f} Nm  "
      f"飽和 (>2.985Nm) {100*np.mean(tau > 2.985):.1f}%")
print(f"  関節速度 最大 {jv.max():.1f} rad/s  99%点 {np.percentile(jv, 99):.1f} "
      f"(DS3235 無負荷 6.0-8.2)")
print(f"  PWM 飽和 {100*pwm_sat.mean():.2f}%  "
      f"4節リンク実現不能 {100*np.mean(inf > 1e-6):.2f}% (最大 {inf.max():.3f} rad)")
report["hardware"] = dict(
    tau_max=float(tau.max()), tau_mean=float(tau.mean()),
    tau_sat_frac=float(np.mean(tau > 2.985)), qvel_max=float(jv.max()),
    qvel_p99=float(np.percentile(jv, 99)), pwm_sat_frac=float(pwm_sat.mean()),
    linkage_infeasible_frac=float(np.mean(inf > 1e-6)),
    linkage_infeasible_max=float(inf.max()))

# =============================================== 出力
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
tt = np.arange(T) * DT
for c, col in ((0, "tab:red"), (1, "tab:orange"), (2, "tab:green")):
    s = cat == c
    if s.sum():
        ax[0].plot(tt, up[:, s].mean(1), lw=2, color=col,
                   label=f"{['flipped','side','upright'][c]} (n={s.sum()})")
ax[0].axhline(rc.upright_cos, ls=":", color="gray")
ax[0].set_title("uprightness vs time"); ax[0].set_xlabel("s"); ax[0].set_ylim(-1.05, 1.05)
ax[0].legend(fontsize=8)
if success.any():
    ax[1].hist(t_rec[success], bins=np.linspace(0, T * DT, 33), color="tab:blue")
ax[1].set_title(f"recovery time [s]  (success {100*success.mean():.0f}%)")
ax[1].set_xlabel("s")
ax[2].plot(tt, 100 * ok.mean(1), lw=2, label="recovery condition met [%]")
ax[2].plot(tt, 100 * (feet.mean(1) / 4), lw=1.5, label="feet on ground [%]")
ax[2].plot(tt, 1000 * hgt.mean(1), lw=1.5, label="base height [mm]")
ax[2].axhline(1000 * consts.STAND_HEIGHT, ls=":", color="gray")
ax[2].set_title("recovery progress"); ax[2].set_xlabel("s"); ax[2].legend(fontsize=8)
for a_ in ax: a_.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "recover.png"), dpi=110)
plt.close(fig)

if not args.no_video:
    import imageio
    # 仰向けから始まって成功した代表を1本。無ければ先頭。
    pick = np.where((cat == 0) & success)[0]
    i = int(pick[np.argsort(t_rec[pick])[len(pick) // 2]]) if len(pick) else 0
    dd = mujoco.MjData(mm)
    frames = []
    with mujoco.Renderer(mm, height=480, width=640) as r:
        for k in range(0, T, 2):
            dd.qpos[:] = qpos[k, i]; dd.qvel[:] = qvels[k, i]
            mujoco.mj_forward(mm, dd)
            r.update_scene(dd, camera="track")
            frames.append(r.render())
    imageio.mimsave(os.path.join(OUT, "recover.mp4"), frames,
                    fps=int(1 / DT / 2), quality=8)
    # コマ送り (復帰の最初の 3秒)
    n = 8
    ks = np.linspace(0, min(T - 1, int(3.0 / DT)), n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    with mujoco.Renderer(mm, height=360, width=360) as r:
        for a_, k in zip(axs, ks):
            dd.qpos[:] = qpos[k, i]; dd.qvel[:] = qvels[k, i]
            mujoco.mj_forward(mm, dd)
            r.update_scene(dd, camera="track")
            a_.imshow(r.render()); a_.axis("off")
            a_.set_title(f"{k*DT:.1f}s", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "recover_filmstrip.png"), dpi=110)
    plt.close(fig)

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(f"\n-> {OUT}/recover.png, report.json"
      f"{'' if args.no_video else ', recover.mp4, recover_filmstrip.png'}")
