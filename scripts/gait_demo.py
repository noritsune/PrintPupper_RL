"""CHAMP の歩容パラメータをそのまま使ったスクリプト歩行 (トロット)。

RL より先にこれを動かす理由:
  学習前に「このモデルは足を振れば前に進むのか」を確認しておく。ここで転ぶなら
  問題はモデル側であって報酬設計ではない、と切り分けられる。
"""
import argparse, os, sys
import numpy as np, mujoco, imageio

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")

L1, L2 = 0.11, 0.12
NOMINAL_H, COM_X, SWING_H = 0.1626, 0.0071, 0.055
STANCE_FRAC = 0.6          # gait.yaml stance_duration/cycle 相当
LEGS = ["lf", "rf", "lh", "rh"]
PHASE = {"lf": 0.0, "rf": 0.5, "lh": 0.5, "rh": 0.0}   # トロット(対角同相)


def ik(x, z):
    """足先 (x, z) [股関節相対] -> (upper, lower) 関節角。

    MuJoCo の +y 軸回り正回転は (0,0,-L) を (-L sin th, 0, -L cos th) に写す。
    つまり関節角が正だと足先は後方(-x)へ動く。順運動学は
        x = -(L1 sin t1 + L2 sin(t1+t2)),  z = -(L1 cos t1 + L2 cos(t1+t2))
    基準姿勢 (t1=+45deg, t2=-90deg) -> (x, z) = (+0.0071, -0.1626) で
    gait.yaml の com_x_translation / nominal_height と符号込みで一致する。
    """
    r2 = x * x + z * z
    c = np.clip((r2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    th2 = -np.arccos(c)                       # knee_orientation ">>" 側の解
    alpha = np.arctan2(-x, -z)
    th1 = alpha - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
    return th1, th2


def foot_traj(p, stride, lift, duty=STANCE_FRAC, h=NOMINAL_H):
    """位相 p in [0,1) -> 足先 (x, z)。stance は後方へ等速、swing は前方へ放物線。"""
    if p < duty:                             # 接地: 後ろへ送る
        s = p / duty
        return stride * (0.5 - s), -h
    s = (p - duty) / (1 - duty)              # 遊脚: 前へ戻しつつ持ち上げる
    return stride * (s - 0.5), -h + lift * np.sin(np.pi * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.15, help="前進速度 [m/s]")
    ap.add_argument("--freq", type=float, default=2.0, help="歩容周波数 [Hz]")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "gait_demo.mp4"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--kp", type=float, default=None, help="サーボ P ゲイン上書き")
    ap.add_argument("--forcerange", type=float, default=None)
    ap.add_argument("--stride", type=float, default=0.07, help="歩幅 [m]")
    ap.add_argument("--height", type=float, default=0.163, help="指令立位高さ [m]")
    ap.add_argument("--lift", type=float, default=SWING_H)
    ap.add_argument("--duty", type=float, default=0.5)
    a = ap.parse_args()

    stride = a.stride if a.stride is not None else a.vx * (a.duty / a.freq)
    lift = a.lift

    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "model", "pupper.xml"))
    if a.kp is not None:
        m.actuator_gainprm[:, 0] = a.kp
        m.actuator_biasprm[:, 1] = -a.kp
        m.actuator_biasprm[:, 2] = -2.0 * np.sqrt(a.kp * m.dof_armature[6])  # dampratio=1
    if a.forcerange is not None:
        m.actuator_forcerange[:] = [-a.forcerange, a.forcerange]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    act = {f"{l}_{j}_joint": m.actuator(f"{l}_{j}_joint").id
           for l in LEGS for j in ("hip", "upper_leg", "lower_leg")}

    r = frames = None
    if not a.no_video:
        r = mujoco.Renderer(m, 480, 640)
        opt = mujoco.MjvOption(); opt.geomgroup[2] = 1; opt.geomgroup[3] = 0
        cams = []
        for az, el, dist in ((90, -8, 0.8), (125, -20, 1.0)):
            c = mujoco.MjvCamera(); c.azimuth = az; c.elevation = el; c.distance = dist
            cams.append(c)
        frames = []

    n = int(a.secs / m.opt.timestep)
    ramp_n = int(1.0 / m.opt.timestep)      # 1s かけて歩容を立ち上げる
    log = []
    for k in range(n):
        t = k * m.opt.timestep
        g = min(1.0, k / ramp_n)            # gain ramp
        for l in LEGS:
            p = (t * a.freq + PHASE[l]) % 1.0
            fx, fz = foot_traj(p, stride * g, lift * g, a.duty, a.height)
            th1, th2 = ik(fx + COM_X, fz)
            d.ctrl[act[f"{l}_hip_joint"]] = 0.0
            d.ctrl[act[f"{l}_upper_leg_joint"]] = th1
            d.ctrl[act[f"{l}_lower_leg_joint"]] = th2
        mujoco.mj_step(m, d)
        log.append((t, d.qpos[0], d.qpos[1], d.qpos[2], *d.qvel[:3]))
        if r is not None and k % 5 == 0:
            tiles = []
            for c in cams:
                c.lookat[:] = [d.qpos[0], d.qpos[1], 0.10]
                r.update_scene(d, c, opt); tiles.append(r.render().copy())
            frames.append(np.hstack(tiles))

    log = np.array(log)
    warm = log[log[:, 0] > 2.0]
    print(f"  前進距離        {log[-1,1]-log[0,1]:+.3f} m  ({a.secs:.0f}s)")
    print(f"  平均前進速度    {np.mean(warm[:,4]):+.3f} m/s  (指令 {a.vx:+.3f})")
    print(f"  横滑り          {log[-1,2]-log[0,2]:+.3f} m")
    print(f"  胴体高さ 平均   {np.mean(warm[:,3]):.4f} m  (基準 {NOMINAL_H+0.012:.4f})")
    print(f"  転倒            {'YES' if log[-1,3] < 0.08 else 'no'}")
    return dict(dist=log[-1,1]-log[0,1], vx=np.mean(warm[:,4]),
                h=np.mean(warm[:,3]), fell=log[-1,3] < 0.08)
    if r is not None:
        imageio.mimsave(a.out, frames, fps=50, quality=8)
        print(f"  -> {a.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
