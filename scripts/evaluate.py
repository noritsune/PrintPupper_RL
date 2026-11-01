"""Phase 6: 学習済みポリシーの評価。

観点:
  1. コマンド追従     指令速度 vs 実測速度
  2. 実機実現可能性   PWM飽和 / 4節リンク / DS3235 のトルク・速度限界
  3. 外乱ロバスト性   押し / 摩擦 / 積載 / サーボ弱体化
  4. 歩容解析         接地パターン / デューティ比 / 足上げ高さ
  5. CHAMP 相当のスクリプト歩容との比較

2 が sim-to-real の関門。ここが通らないポリシーは実機に載せても動かない。
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--run", default=None, help="out/train/<run>")
ap.add_argument("--steps", type=int, default=500, help="1 条件あたりのステップ数")
ap.add_argument("--warmup", type=int, default=150, help="評価から外す立ち上がり")
ap.add_argument("--out", default=None)
ap.add_argument("--no-video", action="store_true")
args = ap.parse_args()

import jax, jax.numpy as jp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mujoco import mjx

from env import constants as consts
from env.deploy import joint_to_servo, pwm_saturated, servo_to_pwm_us
from env.joystick import PupperJoystick, default_config
from env.randomize import domain_randomize
from policy import load
import knee_linkage as kl

if args.run is None and args.ckpt is None:
    runs = sorted(os.listdir(os.path.join(ROOT, "out", "train")))
    args.run = runs[-1]
ckpt = args.ckpt or os.path.join(ROOT, "out", "train", args.run, "ckpt", "final.pkl")
OUT = args.out or os.path.join(ROOT, "out", "eval", args.run or "ckpt")
os.makedirs(OUT, exist_ok=True)
print(f"[eval] {ckpt}\n[out ] {OUT}\n")

env, inference, params = load(ckpt)
env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
DT = env.dt
report = {}

# ---------------------------------------------------------------- 並列ロールアウト
def batch_rollout(commands, n=None, warm=None, model=None, in_axes=None,
                  push_fn=None, seed=0):
    """コマンドの配列を一度に回す。(B,3) -> 記録 dict。"""
    n = n or args.steps
    warm = args.warmup if warm is None else warm
    cmds = jp.asarray(np.atleast_2d(commands), jp.float32)
    B = cmds.shape[0]
    e = env
    if model is not None:
        e = PupperJoystick(config=default_config())
        e._mjx_model = model
        reset = jax.jit(jax.vmap(e.reset, in_axes=(0,)))
        step = jax.jit(jax.vmap(e.step, in_axes=(0, 0)))
    else:
        reset = jax.jit(jax.vmap(env.reset))
        step = jax.jit(jax.vmap(env.step))
    st = reset(jax.random.split(jax.random.PRNGKey(seed), B))
    rec = {k: [] for k in ("v", "w", "q", "dq", "tau", "contact", "footz",
                           "act", "z", "grav", "done")}
    for k in range(n):
        st.info["command"] = cmds
        if push_fn is not None:
            dv = push_fn(k)
            if dv is not None:
                st = st.replace(data=st.data.replace(
                    qvel=st.data.qvel.at[:, :3].add(jp.asarray(dv))))
        a, _ = inference(st.obs, jax.random.PRNGKey(k))
        st = step(st, a)
        d = st.data
        rec["v"].append(d.sensordata[:, env._local_v_adr:env._local_v_adr + 3])
        rec["w"].append(d.sensordata[:, env._gyro_adr:env._gyro_adr + 3])
        rec["q"].append(d.qpos[:, 7:]); rec["dq"].append(d.qvel[:, 6:])
        rec["tau"].append(d.actuator_force); rec["act"].append(a)
        rec["contact"].append(d.sensordata[:, env._feet_contact_adr] > 0)
        rec["footz"].append(d.site_xpos[:, env._feet_site_id, 2])
        rec["z"].append(d.qpos[:, 2])
        rec["grav"].append(-d.sensordata[:, env._up_adr:env._up_adr + 3])
        rec["done"].append(st.done)
    out = {k: np.asarray(jp.stack(v)) for k, v in rec.items()}   # (T,B,...)
    out["warm"] = warm
    out["cmd"] = np.asarray(cmds)
    return out


# センサアドレスを控える (vmap 後に mj_model 参照を避けるため)
mm = env.mj_model
env._local_v_adr = mm.sensor_adr[mm.sensor(consts.LOCAL_LINVEL_SENSOR).id]
env._gyro_adr = mm.sensor_adr[mm.sensor(consts.GYRO_SENSOR).id]
env._up_adr = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id]

# ================================================================ 1. コマンド追従
print("=" * 72)
print("1. コマンド追従")
print("=" * 72)
VX = np.linspace(consts.CMD_VX[0], consts.CMD_VX[1], 9)
VY = np.linspace(consts.CMD_VY[0], consts.CMD_VY[1], 9)
WZ = np.linspace(consts.CMD_WZ[0], consts.CMD_WZ[1], 9)
cmds = np.concatenate([
    np.stack([VX, np.zeros(9), np.zeros(9)], 1),
    np.stack([np.zeros(9), VY, np.zeros(9)], 1),
    np.stack([np.zeros(9), np.zeros(9), WZ], 1)])
R = batch_rollout(cmds)
w = slice(R["warm"], None)
ach_v = R["v"][w].mean(0)      # (B,3)
ach_w = R["w"][w].mean(0)
fell = R["done"].max(0) > 0

track = {}
for i, (lbl, idx, cm, ac) in enumerate([
        ("vx [m/s]", slice(0, 9), VX, ach_v[0:9, 0]),
        ("vy [m/s]", slice(9, 18), VY, ach_v[9:18, 1]),
        ("wz [rad/s]", slice(18, 27), WZ, ach_w[18:27, 2])]):
    err = np.abs(ac - cm)
    gain = np.polyfit(cm, ac, 1)[0]
    track[lbl] = dict(cmd=cm.tolist(), achieved=ac.tolist(),
                      mae=float(err.mean()), gain=float(gain),
                      fell=int(fell[idx].sum()))
    print(f"  {lbl:12s} 傾き {gain:5.2f} (1.0 が理想)  平均絶対誤差 {err.mean():.3f}  "
          f"転倒 {int(fell[idx].sum())}/9")
    print(f"    指令 {np.round(cm,2)}")
    print(f"    実測 {np.round(ac,3)}")
report["tracking"] = track

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (lbl, d) in zip(ax, track.items()):
    a.plot(d["cmd"], d["cmd"], "k--", lw=1, label="ideal")
    a.plot(d["cmd"], d["achieved"], "o-", lw=2, label="achieved")
    a.set_xlabel(f"commanded {lbl}"); a.set_ylabel(f"achieved {lbl}")
    a.set_title(f"{lbl}  gain={d['gain']:.2f}  MAE={d['mae']:.3f}")
    a.grid(alpha=0.3); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracking.png"), dpi=110)
plt.close(fig)

# ================================================== 2. 実機実現可能性 (sim-to-real)
print()
print("=" * 72)
print("2. 実機実現可能性 (DS3235 + 4節リンク + PWM)")
print("=" * 72)
q = R["q"][w].reshape(-1, 12)
dq = R["dq"][w].reshape(-1, 12)
tau = R["tau"][w].reshape(-1, 12)
act = R["act"][w]

STALL = float(mm.actuator_forcerange[0, 1])
NOLOAD = STALL / float(mm.dof_damping[6])
sub = q[:: max(1, len(q) // 4000)]
servo = np.stack([joint_to_servo(x) for x in sub])
sat = pwm_saturated(servo)
us = servo_to_pwm_us(servo)

infeas = np.array([
    any(kl.compute(a_, b_) is None or not kl.servo_cmd_ok(kl.compute(a_, b_), "lower")
        for a_, b_ in zip(x.reshape(4, 3)[:, 1], x.reshape(4, 3)[:, 2]))
    for x in sub])

feas = {
    "pwm_saturation_pct": float(sat.mean() * 100),
    "pwm_min_us": float(us.min()), "pwm_max_us": float(us.max()),
    "linkage_infeasible_pct": float(infeas.mean() * 100),
    "torque_p50": float(np.percentile(np.abs(tau), 50)),
    "torque_p99": float(np.percentile(np.abs(tau), 99)),
    "torque_max": float(np.abs(tau).max()),
    "torque_stall": STALL,
    "torque_util_p99_pct": float(np.percentile(np.abs(tau), 99) / STALL * 100),
    "speed_p99": float(np.percentile(np.abs(dq), 99)),
    "speed_max": float(np.abs(dq).max()),
    "speed_noload": NOLOAD,
    "speed_util_p99_pct": float(np.percentile(np.abs(dq), 99) / NOLOAD * 100),
    "action_rate_p99": float(np.percentile(np.abs(np.diff(act, axis=0)), 99)),
}
print(f"  PWM 飽和            {feas['pwm_saturation_pct']:6.2f} %   "
      f"パルス幅 {feas['pwm_min_us']:.0f}-{feas['pwm_max_us']:.0f} us (限界 500-2500)")
print(f"  4節リンク実現不能   {feas['linkage_infeasible_pct']:6.2f} %")
print(f"  トルク  中央 {feas['torque_p50']:.3f} / 99%点 {feas['torque_p99']:.3f} / "
      f"最大 {feas['torque_max']:.3f} Nm   (DS3235 stall {STALL:.1f})")
print(f"          99%点はストールの {feas['torque_util_p99_pct']:.1f} %")
print(f"  角速度  99%点 {feas['speed_p99']:.2f} / 最大 {feas['speed_max']:.2f} rad/s   "
      f"(無負荷 {NOLOAD:.1f})")
print(f"          99%点は無負荷速度の {feas['speed_util_p99_pct']:.1f} %")
print(f"  指令変化率 99%点    {feas['action_rate_p99']:.3f} (action 単位/20ms)")
report["feasibility"] = feas

gates = [
    ("PWM 飽和 < 1%", feas["pwm_saturation_pct"] < 1.0),
    ("リンク実現不能 < 1%", feas["linkage_infeasible_pct"] < 1.0),
    ("トルク99%点 < ストールの70%", feas["torque_util_p99_pct"] < 70),
    ("角速度99%点 < 無負荷の80%", feas["speed_util_p99_pct"] < 80),
]
print()
for name, ok in gates:
    print(f"  [{'OK  ' if ok else 'NG  '}] {name}")
report["gates"] = {n: bool(o) for n, o in gates}

# ================================================================ 3. 歩容解析
print()
print("=" * 72)
print("3. 歩容解析 (前進 0.2 m/s)")
print("=" * 72)
G = batch_rollout(np.array([[0.20, 0, 0]]), n=600)
gw = slice(G["warm"], None)
c = G["contact"][gw, 0]                 # (T,4)
fz = G["footz"][gw, 0] - consts.FOOT_RADIUS
duty = c.mean(0)
# 歩容周期: 左前脚の接地立ち上がりから推定
rise = np.flatnonzero(np.diff(c[:, 0].astype(int)) > 0)
period = float(np.mean(np.diff(rise)) * DT) if len(rise) > 2 else float("nan")
swing_peak = np.array([fz[:, i][~c[:, i]].max() if (~c[:, i]).any() else 0.0
                       for i in range(4)])
# 対角脚の位相差 (トロットなら 0.5)
def phase_of(i):
    r = np.flatnonzero(np.diff(c[:, i].astype(int)) > 0)
    return r[0] if len(r) else 0
ph = np.array([phase_of(i) for i in range(4)]) * DT
gait = dict(duty_factor=duty.tolist(), period_s=period,
            freq_hz=float(1 / period) if period == period else float("nan"),
            swing_peak_m=swing_peak.tolist(),
            swing_target_m=consts.SWING_HEIGHT,
            body_height_m=float(G["z"][gw, 0].mean()),
            pitch_roll_deg=[float(np.degrees(np.arcsin(np.clip(
                G["grav"][gw, 0, i], -1, 1))).std()) for i in (0, 1)])
print(f"  接地デューティ比  {np.round(duty,3)}  (脚順 lf, rf, lh, rh)")
print(f"  歩容周期          {period:.3f} s  = {gait['freq_hz']:.2f} Hz")
print(f"  遊脚の最高到達点  {np.round(swing_peak*1000,1)} mm  "
      f"(gait.yaml swing_height {consts.SWING_HEIGHT*1000:.0f} mm)")
print(f"  胴体高さ          {gait['body_height_m']:.4f} m "
      f"(基準 {consts.STAND_HEIGHT+consts.FOOT_RADIUS:.4f})")
print(f"  接地立ち上がり時刻 {np.round(ph,3)} s -> トロットなら対角が同相")
report["gait"] = gait

fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
t = np.arange(c.shape[0]) * DT
for i, nm in enumerate(consts.LEGS):
    ax[0].fill_between(t, i, i + 0.8, where=c[:, i], step="mid", alpha=0.8)
    ax[1].plot(t, fz[:, i] * 1000, lw=1, label=nm)
ax[0].set_yticks(np.arange(4) + 0.4); ax[0].set_yticklabels(consts.LEGS)
ax[0].set_title("contact (stance) pattern @ vx=0.20 m/s")
ax[1].axhline(consts.SWING_HEIGHT * 1000, color="k", ls="--", lw=1,
              label="swing_height")
ax[1].set_ylabel("foot height [mm]"); ax[1].set_xlabel("time [s]")
ax[1].legend(fontsize=7, ncol=5); ax[0].grid(alpha=0.3); ax[1].grid(alpha=0.3)
ax[1].set_xlim(0, min(3.0, t[-1]))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gait.png"), dpi=110)
plt.close(fig)

# ================================================================ 4. ロバスト性
print()
print("=" * 72)
print("4. 外乱ロバスト性 (前進 0.2 m/s, 各16試行)")
print("=" * 72)
rob = {}
base_cmd = np.tile([0.20, 0, 0], (16, 1))

# (a) 横方向の押し
for mag in (0.3, 0.6, 1.0, 1.5):
    def pf(k, mag=mag):
        if k == 250:
            d = np.zeros((16, 3)); d[:, 1] = mag * np.where(np.arange(16) % 2, 1, -1)
            return d
        return None
    Rp = batch_rollout(base_cmd, n=450, push_fn=pf, seed=5)
    surv = float((Rp["done"].max(0) == 0).mean())
    rob[f"push_{mag}"] = surv
    print(f"  横押し {mag:4.1f} m/s     生存 {surv*100:5.1f} %")

# (b) 摩擦
for fr in (0.3, 0.5, 0.8, 1.2):
    m2, ax2 = domain_randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(1), 16))
    m2 = m2.tree_replace({"geom_friction": m2.geom_friction.at[:, :, 0].set(fr)})
    Rf = batch_rollout(base_cmd, n=400, model=m2, in_axes=ax2, seed=6)
    ww = slice(Rf["warm"], None)
    rob[f"friction_{fr}"] = dict(
        survive=float((Rf["done"].max(0) == 0).mean()),
        vx=float(Rf["v"][ww, :, 0].mean()))
    print(f"  摩擦 {fr:4.1f}          生存 {rob[f'friction_{fr}']['survive']*100:5.1f} %  "
          f"実測 vx {rob[f'friction_{fr}']['vx']:+.3f} m/s")

# (c) 積載 (胴体質量を増やす)
for extra in (0.0, 0.2, 0.4, 0.6):
    mm2 = env.mjx_model
    bm = mm2.body_mass.at[1].add(extra)
    m3 = mm2.tree_replace({"body_mass": jp.broadcast_to(bm, (16,) + bm.shape)})
    ax3 = jax.tree_util.tree_map(lambda x: None, mm2).tree_replace({"body_mass": 0})
    Rl = batch_rollout(base_cmd, n=400, model=m3, in_axes=ax3, seed=7)
    ww = slice(Rl["warm"], None)
    rob[f"payload_{extra}"] = dict(
        survive=float((Rl["done"].max(0) == 0).mean()),
        vx=float(Rl["v"][ww, :, 0].mean()))
    print(f"  積載 +{extra:.1f} kg      生存 {rob[f'payload_{extra}']['survive']*100:5.1f} %  "
          f"実測 vx {rob[f'payload_{extra}']['vx']:+.3f} m/s")

# (d) サーボ弱体化 (電池消耗)
for fr in (3.0, 2.4, 1.8, 1.2):
    mm2 = env.mjx_model
    f2 = mm2.actuator_forcerange.at[:, 0].set(-fr).at[:, 1].set(fr)
    m4 = mm2.tree_replace({"actuator_forcerange":
                           jp.broadcast_to(f2, (16,) + f2.shape)})
    ax4 = jax.tree_util.tree_map(lambda x: None, mm2).tree_replace(
        {"actuator_forcerange": 0})
    Rs = batch_rollout(base_cmd, n=400, model=m4, in_axes=ax4, seed=8)
    ww = slice(Rs["warm"], None)
    rob[f"stall_{fr}"] = dict(
        survive=float((Rs["done"].max(0) == 0).mean()),
        vx=float(Rs["v"][ww, :, 0].mean()))
    print(f"  ストール {fr:4.1f} Nm    生存 {rob[f'stall_{fr}']['survive']*100:5.1f} %  "
          f"実測 vx {rob[f'stall_{fr}']['vx']:+.3f} m/s")
report["robustness"] = rob

fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
ax[0].bar([str(m) for m in (0.3, 0.6, 1.0, 1.5)],
          [rob[f"push_{m}"] * 100 for m in (0.3, 0.6, 1.0, 1.5)])
ax[0].set_title("survival vs lateral push [m/s]")
for a, (key, xs, lbl) in zip(ax[1:], [
        ("friction", (0.3, 0.5, 0.8, 1.2), "ground friction"),
        ("payload", (0.0, 0.2, 0.4, 0.6), "extra payload [kg]"),
        ("stall", (3.0, 2.4, 1.8, 1.2), "servo stall torque [Nm]")]):
    a2 = a.twinx()
    a.bar([str(x) for x in xs], [rob[f"{key}_{x}"]["survive"] * 100 for x in xs],
          alpha=0.65)
    a2.plot([str(x) for x in xs], [rob[f"{key}_{x}"]["vx"] for x in xs],
            "o-", color="tab:red")
    a2.axhline(0.20, color="k", ls="--", lw=1)
    a.set_title(lbl); a.set_ylabel("survival %"); a2.set_ylabel("vx [m/s]", color="tab:red")
for a in ax: a.set_ylim(0, 105); a.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "robustness.png"), dpi=110)
plt.close(fig)

# ================================================================ 5. 動画
if not args.no_video:
    print()
    print("=" * 72)
    print("5. 動画")
    print("=" * 72)
    import imageio
    from mujoco_playground._src import mjx_env as _mjx_env
    e1 = PupperJoystick(config=default_config())
    e1._mjx_model = e1._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
    jr, js = jax.jit(e1.reset), jax.jit(e1.step)
    st = jr(jax.random.PRNGKey(0))
    segs = [("前進 0.30", [0.30, 0, 0]), ("後退 0.20", [-0.20, 0, 0]),
            ("横歩き 0.15", [0, 0.15, 0]), ("旋回 1.0", [0, 0, 1.0]),
            ("斜め+旋回", [0.20, 0.10, 0.5]), ("停止", [0, 0, 0])]
    L = 100
    traj = []
    for si, (nm, c) in enumerate(segs):
        for k in range(L):
            st.info["command"] = jp.array(c, jp.float32)
            a, _ = inference(st.obs, jax.random.PRNGKey(si * L + k))
            st = js(st, a); traj.append(st)
    frames = e1.render(traj[::2], height=420, width=640, camera="track")
    p = os.path.join(OUT, "eval.mp4")
    imageio.mimsave(p, frames, fps=25, quality=8)
    print(f"  -> {p}  ({len(frames)} frames, 各コマンド {L*DT:.0f}s)")
    strip = np.hstack([frames[int((i + 0.5) * len(frames) / 6)] for i in range(6)])
    imageio.imwrite(os.path.join(OUT, "eval_strip.png"), strip)
    print(f"  -> {os.path.join(OUT,'eval_strip.png')}  ({', '.join(n for n,_ in segs)})")

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print()
print("=" * 72)
allok = all(report["gates"].values())
print(f"実機実現可能性ゲート: {'全項目 PASS' if allok else '未達あり'}")
print(f"詳細 -> {OUT}")
print("=" * 72)
