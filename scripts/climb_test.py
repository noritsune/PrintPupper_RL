"""段差乗り越え (env/climb.py) の評価。何 mm の段まで乗れるかを直接測る。

step_test.py との違い: あちらは「前進指令で歩いて段を越えられたか」を測る
歩行方策用のベンチで、歩容クロックと速度指令が前提になっている。こちらは
準静的登坂スキル用で、段の正面に置いたところから「機体ごと段の上に乗れたか」
だけを見る。

測るもの:
  1. 登坂成功率   段の上に 4脚で乗り、0.5s 連続でその状態を保てたか
  2. どこで詰まるか  段の上に乗せられた足の最大数 (0-4)。1本も上がらないのか、
                  3本まで上げて重心を送れずに落ちるのかで打つ手が変わる
  3. 登坂時間     成功までの時間
  4. 実機成立性   トルク飽和 / 関節速度 / PWM 飽和 / 4節リンクの実現可能性

使い方:
  python scripts/climb_test.py --run c60
  python scripts/climb_test.py --ckpt out/train/c60/ckpt/final.pkl \
      --heights 40,50,60,70,80 --trials 32
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
ap.add_argument("--heights", default=None,
                help="試す段差高さ [mm] のカンマ区切り。既定は env の帯 (30-70mm)")
ap.add_argument("--trials", type=int, default=24, help="高さあたりの試行数")
ap.add_argument("--secs", type=float, default=10.0)
ap.add_argument("--hold", type=float, default=0.5,
                help="成功と認める連続保持時間 [s]")
ap.add_argument("--no-randomize", action="store_true",
                help="ドメインランダム化を切る (公称モデルだけで測る)")
ap.add_argument("--video", type=float, default=None,
                help="この高さ [mm] の 1 試行を動画にする")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import jax, jax.numpy as jp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx

from env import _compat; _compat.install()
import warp as wp; wp.config.quiet = True
from mujoco_playground._src import wrapper

from env import constants as consts
from env.climb import PupperClimb, climb_config
from env.deploy import joint_to_servo, pwm_saturated
from env.linkage import infeasibility
from env.randomize import domain_randomize
from policy import load

if args.run is None and args.ckpt is None:
    args.run = sorted(os.listdir(os.path.join(ROOT, "out", "train")))[-1]
ckpt = args.ckpt or os.path.join(ROOT, "out", "train", args.run, "ckpt", "latest.pkl")
OUT = args.out or os.path.join(ROOT, "out", "eval", (args.run or "ckpt") + "_climb")
os.makedirs(OUT, exist_ok=True)

cfg = climb_config()
if args.heights is not None:
    cfg.climb.heights = [float(h) / 1000.0 for h in args.heights.split(",")]
HEIGHTS = np.asarray(cfg.climb.heights)
NH = len(HEIGHTS)
B = NH * args.trials
cfg.naconmax = 8 * B
cfg.episode_length = int(round(args.secs / cfg.ctrl_dt))
env = PupperClimb(config=cfg)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
_, inference, _ = load(ckpt, env=env)
DT = env.dt
T = int(round(args.secs / DT))
HOLD = max(1, int(round(args.hold / DT)))
SCALE = np.tile(np.asarray(cfg.climb.action_scale), 4)

print(f"[eval] {ckpt}\n[out ] {OUT}")
print(f"[env ] 段差 {(HEIGHTS*1000).round(0).astype(int).tolist()} mm  "
      f"高さあたり {args.trials} 試行  {args.secs:.0f}s  "
      f"ランダム化 {'なし' if args.no_randomize else 'あり'}")
print(f"[env ] 可動域 (関節別) {list(cfg.climb.action_scale)} rad  "
      f"成功 = 4脚が段の上 & 胴体が +{cfg.climb.success_x*1000:.0f}mm 超 & "
      f"{args.hold:.1f}s 連続\n")

# 帯の割り当て: env i は帯 (i // trials)。env が reset で引いた帯からの差分だけ
# y を動かし、初期姿勢のばらつきはそのまま残す。手前は帯によらず平地 (高さ 0)
# なので、y をずらしても力学は変わらない。mjx.step は qpos から作り直すので
# ここで mjx.forward を掛け直す必要はない (obs も y に依存しない)。
band_i = np.arange(B) // args.trials
BAND_Y = np.asarray(env._band_y)
BAND_H = np.asarray(env._band_h)


def relocate(state):
    dy = jp.asarray(BAND_Y[band_i]) - state.info["band_y"]
    data = state.data.replace(qpos=state.data.qpos.at[:, 1].add(dy))
    info = dict(state.info)
    info["band_y"] = jp.asarray(BAND_Y[band_i])
    info["step_h"] = jp.asarray(BAND_H[band_i])
    key = "AutoResetWrapper_first_data"
    if key in info:            # 自動リセットの戻り先も同じ帯にしておく
        info[key] = info[key].replace(qpos=info[key].qpos.at[:, 1].add(dy))
    return state.replace(data=data, info=info)


# 学習と同じラッパ (ドメインランダム化 + 自動リセット) で回す。打ち切り後の
# ステップは alive で落とすので、自動リセットの中身は結果に入らない。
rand_fn = None if args.no_randomize else functools.partial(
    domain_randomize, rng=jax.random.split(jax.random.PRNGKey(42), B))
wrapped = wrapper.wrap_for_brax_training(
    env, episode_length=cfg.episode_length, action_repeat=1,
    randomization_fn=rand_fn)
w_reset, w_step = jax.jit(wrapped.reset), jax.jit(wrapped.step)

state = relocate(w_reset(jax.random.split(jax.random.PRNGKey(0), B)))

rec = {k: [] for k in ("x", "feet", "success", "done", "tau", "qvel", "qpos",
                       "margin", "act")}
print(f"1. ロールアウト ({T} step = {args.secs:.0f}s)")
for k in range(T):
    act, _ = inference(state.obs, jax.random.PRNGKey(k))
    state = w_step(state, act)
    rec["act"].append(np.asarray(act))
    m = state.metrics
    rec["x"].append(np.asarray(m["clb/x"]))
    rec["feet"].append(np.asarray(m["clb/feet"]))
    rec["success"].append(np.asarray(m["clb/success"]))
    rec["margin"].append(np.asarray(m["clb/margin"]))
    rec["done"].append(np.asarray(state.done))
    rec["tau"].append(np.asarray(state.data.actuator_force))
    rec["qvel"].append(np.asarray(state.data.qvel))
    rec["qpos"].append(np.asarray(state.data.qpos))
rec = {k: np.stack(v) for k, v in rec.items()}          # (T, B, ...)

# 打ち切り後のステップは評価に入れない (done のあとも物理は進み続ける)。
alive = np.cumprod(1.0 - np.clip(rec["done"], 0, 1), axis=0) > 0.5
alive = np.concatenate([np.ones((1, B), bool), alive[:-1]], 0)   # done の瞬間は含む
suc = (rec["success"] > 0.5) & alive

# 0.5s 連続で成功状態を保てたか。畳み込みで連続区間を拾う。
run = np.zeros_like(suc, np.int32)
for t in range(T):
    run[t] = np.where(suc[t], (run[t - 1] if t else 0) + 1, 0)
ok = run >= HOLD
t_climb = np.where(ok.any(0), ok.argmax(0) * DT, np.nan)

print("\n2. 登坂成功率")
print("  段差    成功率    登坂時間 中央   段上の足 最大   到達 x 中央   生存率")
print("  " + "-" * 72)
rows = []
for i, h in enumerate(HEIGHTS):
    s = slice(i * args.trials, (i + 1) * args.trials)
    succ = ok[:, s].any(0)
    tc = t_climb[s]
    feet_max = rec["feet"][:, s].max(0)
    x_end = np.array([rec["x"][:, s][alive[:, s][:, j], j][-1]
                      if alive[:, s][:, j].any() else np.nan
                      for j in range(args.trials)])
    surv = alive[-1, s]
    print(f"  {h*1000:3.0f}mm   {100*succ.mean():5.1f} %      "
          f"{np.nanmedian(tc) if succ.any() else float('nan'):5.2f} s        "
          f"{np.median(feet_max):4.1f} / 4        "
          f"{np.nanmedian(x_end)*1000:+6.0f} mm    {100*surv.mean():5.1f} %")
    rows.append(dict(height_mm=float(h * 1000), success=float(succ.mean()),
                     t_climb_median=float(np.nanmedian(tc)) if succ.any() else None,
                     feet_max_median=float(np.median(feet_max)),
                     x_end_median_mm=float(np.nanmedian(x_end) * 1000),
                     survival=float(surv.mean())))

hit = [r["height_mm"] for r in rows if r["success"] >= 0.5]
print(f"\n  -> 越えられる段差 (成功率 50% 以上の最大) = "
      f"{max(hit):.0f} mm" if hit else "\n  -> 成功率 50% を超えた高さは無い")

# ================================================================ 実機成立性
print("\n3. 実機成立性 (全エピソード・全ステップ、打ち切り後は除く)")
mask = alive
tau = np.abs(rec["tau"])[mask]
qv = np.abs(rec["qvel"][..., 6:])[mask]
q = rec["qpos"][..., 7:][mask]
# PWM 飽和は「実機に出す指令」で決まるので、達成された関節角ではなく
# 目標角 (基準姿勢 + 行動 x 関節別スケール) で見る。
targets = (np.asarray(consts.DEFAULT_POSE) + rec["act"] * SCALE)[mask]
sub = np.random.default_rng(0).choice(len(q), size=min(4000, len(q)), replace=False)
inf = np.array([float(infeasibility(jp.asarray(x))) for x in q[sub]])
servo = np.array([joint_to_servo(x) for x in targets[sub]])
pwm_sat = np.array([pwm_saturated(s) for s in servo])
print(f"  トルク 最大 {tau.max():.2f} Nm  平均 {tau.mean():.2f} Nm  "
      f"飽和 (>2.985Nm) {100*np.mean(tau > 2.985):.1f}%  "
      f"軟限界 (>1.5Nm) 超え {100*np.mean(tau > 1.5):.1f}%")
print(f"  関節速度 最大 {qv.max():.1f} rad/s  99%点 {np.percentile(qv, 99):.1f} "
      f"(DS3235 無負荷 6.0-8.2)")
print(f"  PWM 飽和 {100*pwm_sat.mean():.2f}%  4節リンク実現不能 "
      f"{100*np.mean(inf > 1e-6):.2f}% (最大 {inf.max():.3f} rad)")
print(f"  静的安定余裕 中央 {1000*np.median(rec['margin'][mask]):+.1f} mm  "
      f"(正の時間の割合 {100*np.mean(rec['margin'][mask] > 0):.0f}%)")
report = dict(ckpt=ckpt, secs=args.secs, trials=args.trials, hold_s=args.hold,
              action_scale=list(cfg.climb.action_scale), heights=rows,
              hardware=dict(tau_max=float(tau.max()), tau_mean=float(tau.mean()),
                            tau_sat_frac=float(np.mean(tau > 2.985)),
                            qvel_max=float(qv.max()),
                            qvel_p99=float(np.percentile(qv, 99)),
                            pwm_sat_frac=float(pwm_sat.mean()),
                            linkage_infeasible_frac=float(np.mean(inf > 1e-6)),
                            margin_median_mm=float(1000 * np.median(
                                rec["margin"][mask]))))

# ================================================================ 出力
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(HEIGHTS * 1000, [100 * r["success"] for r in rows], width=6,
          color="tab:orange")
ax[0].axhline(50, ls=":", color="gray")
ax[0].set_title("climb success [%]"); ax[0].set_xlabel("step height [mm]")
ax[0].set_ylim(0, 102)
ax[1].bar(HEIGHTS * 1000, [r["feet_max_median"] for r in rows], width=6,
          color="tab:blue")
ax[1].set_title("feet placed on the ledge (median of max)")
ax[1].set_xlabel("step height [mm]"); ax[1].set_ylim(0, 4.2)
t = np.arange(T) * DT
for i, h in enumerate(HEIGHTS):
    s = slice(i * args.trials, (i + 1) * args.trials)
    ax[2].plot(t, np.median(np.where(alive[:, s], rec["x"][:, s], np.nan), 1) * 1000,
               lw=1.5, label=f"{h*1000:.0f}mm")
ax[2].axhline(cfg.climb.success_x * 1000, ls=":", color="gray")
ax[2].set_title("trunk x from the ledge [mm] (median)"); ax[2].set_xlabel("s")
ax[2].legend(fontsize=7)
for a_ in ax: a_.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "climb.png"), dpi=110)
plt.close(fig)

if args.video is not None:
    import imageio
    i = int(np.argmin(np.abs(HEIGHTS * 1000 - args.video)))
    s = slice(i * args.trials, (i + 1) * args.trials)
    j = i * args.trials + int(np.nanargmax(np.where(ok[:, s].any(0), 1, 0)
                                           + rec["feet"][:, s].max(0) / 8))
    mj = env.mj_model
    dd = mujoco.MjData(mj)
    frames = []
    with mujoco.Renderer(mj, height=480, width=640) as r:
        for k in range(0, T, 2):
            dd.qpos[:] = rec["qpos"][k, j]; dd.qvel[:] = rec["qvel"][k, j]
            mujoco.mj_forward(mj, dd)
            r.update_scene(dd, camera="track")
            frames.append(r.render())
    imageio.mimsave(os.path.join(OUT, "climb.mp4"), frames,
                    fps=int(1 / DT / 2), quality=8)

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(f"\n-> {OUT}/climb.png, report.json"
      f"{', climb.mp4' if args.video is not None else ''}")
