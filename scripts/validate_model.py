"""Phase 3-4 物理検証。ここを飛ばすと Phase 5 で最悪のデバッグに入る。"""
import numpy as np, mujoco, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from knee_linkage import compute, servo_cmd_ok, NEUTRAL, SERVO_HALF_RANGE

M = os.path.join(os.path.dirname(__file__), "..", "model", "pupper.xml")
m = mujoco.MjModel.from_xml_path(M)
d = mujoco.MjData(m)
G = 9.81
ok = lambda c: "OK  " if c else "NG  "
fails = []
def check(cond, label):
    print(f"  [{ok(cond)}] {label}")
    if not cond: fails.append(label)

print("=" * 70)
print("1. モデル構造")
print("=" * 70)
print(f"  nq={m.nq} nv={m.nv} njnt={m.njnt} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
check(m.nu == 12, "アクチュエータ 12軸")
check(m.nq == 19, "nq=19 (freejoint 7 + 12関節)")

print()
print("=" * 70)
print("2. 質量・慣性")
print("=" * 70)
total = m.body_mass.sum()
print(f"  {'body':22s} {'mass[kg]':>9s} {'Ixx':>10s} {'Iyy':>10s} {'Izz':>10s}")
for i in range(1, m.nbody):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    I = m.body_inertia[i]
    print(f"  {n:22s} {m.body_mass[i]:9.4f} {I[0]:10.3e} {I[1]:10.3e} {I[2]:10.3e}")
print(f"  {'TOTAL':22s} {total:9.4f}")
check(1.2 <= total <= 2.0, f"総質量 {total:.3f} kg が Pupper 級 1.2-2.0kg の範囲")
bi = m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")].max()
worst = max(m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_{p}_link")].max()
            for l in ("lf","rf","lh","rh") for p in ("hip","upper_leg","lower_leg"))
print(f"  胴体最大慣性 {bi:.3e} / 脚リンク最大慣性 {worst:.3e}")
check(worst < bi, "脚リンク慣性が胴体より小さい (逆転なし)")
for i in range(1, m.nbody):
    Ix, Iy, Iz = m.body_inertia[i]
    if not (Ix+Iy >= Iz-1e-12 and Iy+Iz >= Ix-1e-12 and Iz+Ix >= Iy-1e-12):
        fails.append(f"三角不等式違反: body {i}")
check(not any("三角不等式" in f for f in fails), "全リンクが慣性三角不等式を満たす")

print()
print("=" * 70)
print("3. 基準姿勢と gait.yaml の整合")
print("=" * 70)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{l}_foot") for l in ("lf","rf","lh","rh")]
fz = np.array([d.site_xpos[f][2] for f in feet])
base_z = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,"base_link")][2]
h = base_z - fz.mean()
print(f"  胴体高さ - 足先高さ = {h:.4f} m   (gait.yaml nominal_height = 0.1626)")
check(abs(h - 0.1626) < 0.002, "立位高さが nominal_height と一致")
fx = np.array([d.site_xpos[f][0] for f in feet])
hipx = np.array([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,f"{l}_hip_link")][0] for l in ("lf","rf","lh","rh")])
print(f"  足先の股関節からの前後ずれ = {np.round(fx-hipx,4)} m  (com_x_translation = 0.0071)")
check(np.allclose(np.abs(fx-hipx), 0.0071, atol=0.001), "足先前後位置が com_x_translation と一致")
# 足先 site は接触球の中心なので、接地時の site z は球半径に一致するのが正
print(f"  足先 site 高さ (床=0)     = {np.round(fz,5)} m  (足球半径 0.012 に一致すべき)")
check(np.allclose(fz, 0.012, atol=1e-3), "基準姿勢で足球が床に接している")

print()
print("=" * 70)
print("4. 関節リミット vs 実サーボ可動域")
print("=" * 70)
print(f"  {'joint':26s} {'MJCF range[deg]':>22s} {'実サーボ由来[deg]':>22s}")
exp = {"hip_joint": (-90, 90), "upper_leg_joint": (-45, 135)}
for i in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if n == "root": continue
    lo, hi = np.degrees(m.jnt_range[i])
    key = n[3:]
    e = exp.get(key)
    if e:
        if not (abs(lo-e[0]) < 0.1 and abs(hi-e[1]) < 0.1): fails.append(f"{n} range mismatch")
        if n.startswith("lf"): print(f"  {n:26s} [{lo:7.1f},{hi:7.1f}] [{e[0]:7.1f},{e[1]:7.1f}]")
    elif n.startswith("lf"):
        print(f"  {n:26s} [{lo:7.1f},{hi:7.1f}] {'4節リンク連成(包絡線)':>22s}")
check(not any("range mismatch" in f for f in fails), "hip/upper のリミットが PWM 可動域と一致")

print()
print("=" * 70)
print("5. サーボ (DS3235) のモデル検証")
print("=" * 70)
STALL = float(m.actuator_forcerange[0, 1])
mujoco.mj_resetDataKeyframe(m, d, 0)
for _ in range(1500): mujoco.mj_step(m, d)   # 6s 静定
tau = np.abs(d.actuator_force)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print(f"  ストールトルク設定 = {STALL:.2f} Nm  (DS3235: 30kg-cm@6V=2.94 / 35kg-cm@6.8V=3.43)")
for i in range(3):
    print(f"  静止保持 {names[i]:24s} |tau| = {tau[i]:.4f} Nm")
mx = tau.max()
print(f"  最大 |tau| = {mx:.4f} Nm  ({mx/STALL*100:.1f}% of stall)")
check(mx < STALL * 0.5, "静止保持トルクがストールの50%未満 (DS3235 なら大きな余裕)")

# トルク-速度特性: 無負荷で関節を振り切らせ、到達角速度が公称に収まるか
m2 = mujoco.MjModel.from_xml_path(M)
for i in range(m2.nbody):        # 重力と負荷を消して純粋な無負荷応答を見る
    pass
m2.opt.gravity[:] = 0
d2 = mujoco.MjData(m2)
mujoco.mj_resetDataKeyframe(m2, d2, 0)
jid = m2.joint("lf_upper_leg_joint").id
aid = m2.actuator("lf_upper_leg_joint").id
d2.ctrl[:] = m2.keyframe("stand").ctrl
d2.ctrl[aid] = m2.jnt_range[jid][1]          # 上限まで全力で振る
vmax = 0.0
for _ in range(500):
    mujoco.mj_step(m2, d2)
    vmax = max(vmax, abs(d2.qvel[m2.jnt_dofadr[jid]]))
NOLOAD = STALL / float(m2.dof_damping[m2.jnt_dofadr[jid]])
print(f"  無負荷での最大到達角速度 = {vmax:.2f} rad/s")
print(f"  モデル上の無負荷速度     = {NOLOAD:.2f} rad/s "
      f"(DS3235: 6.16@4.8V - 8.05@6.8V rad/s)")
check(vmax <= NOLOAD * 1.15, "到達角速度が無負荷速度を超えない (トルク-速度特性が効いている)")
check(6.0 <= NOLOAD <= 8.2, "モデルの無負荷速度が DS3235 の実測レンジ内")

print()
print("=" * 70)
print("6. 立位安定性 (6秒静定後)")
print("=" * 70)
base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
print(f"  胴体位置 = {np.round(d.xpos[base],4)}  (初期 z={0.1626+0.012:.4f})")
print(f"  胴体速度 = {np.linalg.norm(d.qvel[:3]):.5f} m/s  角速度 {np.linalg.norm(d.qvel[3:6]):.5f} rad/s")
sag = (0.1626+0.012) - d.xpos[base][2]
print(f"  沈み込み = {sag*1000:+.1f} mm  (kp={m.actuator_gainprm[0,0]:.0f} のサーボ撓み)")
check(np.linalg.norm(d.qvel[:6]) < 0.01, "静止している (発振・滑走なし)")
check(abs(sag) < 0.03, "沈み込みが 30mm 未満")
check(np.linalg.norm(d.xpos[base][:2]) < 0.02, "水平ドリフトなし")

print()
print("=" * 70)
print("7. 落下テスト (0.30m から自由落下)")
print("=" * 70)
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[2] = 0.30
mujoco.mj_forward(m, d)
zs, contacts = [], []
for k in range(2000):   # 8 s
    mujoco.mj_step(m, d)
    zs.append(d.xpos[base][2]); contacts.append(d.ncon)
zs = np.array(zs)
land = int(np.argmax(np.array(contacts) > 0))
print(f"  着地 t={land*0.004:.2f}s  最低点 z={zs.min():.4f}  終端 z={zs[-1]:.4f}")
bounce = zs[land:].max() - zs[land:].min()
print(f"  着地後の上下動幅 = {bounce*1000:.1f} mm")
print(f"  終端速度 = {np.linalg.norm(d.qvel[:3]):.5f} m/s")
check(np.linalg.norm(d.qvel[:6]) < 0.05, "着地後に静止する (跳ね続けない)")
check(bounce < 0.05, "跳ね返りが 50mm 未満")
check(np.isfinite(zs).all() and zs.min() > -0.05, "床を貫通しない / 発散しない")

print()
print("=" * 70)
print("8. 4節リンク膝の実現可能性 (MJCF リミット内の全域)")
print("=" * 70)
jl = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): m.jnt_range[i] for i in range(m.njnt)}
u_lo, u_hi = jl["lf_upper_leg_joint"]; l_lo, l_hi = jl["lf_lower_leg_joint"]
U = np.linspace(u_lo, u_hi, 60); L = np.linspace(l_lo, l_hi, 60)
n_ok = sum(1 for a in U for b in L
           if (r := compute(a, b)) is not None and servo_cmd_ok(r, "lower"))
frac = n_ok / (len(U)*len(L))
print(f"  MJCF リミット矩形のうち実機で実現可能な割合 = {frac*100:.1f}%")
print(f"  -> 残り {100-frac*100:.1f}% は Phase 4 のリンク実現可能性ペナルティで排除する")
r0 = compute(np.pi/4, -np.pi/2)
print(f"  基準姿勢の膝サーボ角 = {np.degrees(r0):.2f} deg (サーボ neutral -45.0 deg)")
check(abs(np.degrees(r0) - (-45.0)) < 3.0, "基準姿勢が膝サーボの中立付近")

print()
print("=" * 70)
print(("検証 NG: " + "; ".join(fails)) if fails else "全検証項目 PASS")
print("=" * 70)
sys.exit(1 if fails else 0)
