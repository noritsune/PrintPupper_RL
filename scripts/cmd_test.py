"""速度指令にどれだけ追従できるかを、指令の向きごとに測る。

これまで歩行の成績は `scripts/step_test.py` の「前進して段を越えられたか」
だけで見ていた。前進だけが得意な方策でもその数字は出るので、**横歩きと旋回に
穴があっても気付けない**。ここでは指令を 1 つずつ固定して 8 秒歩かせ、
実際に出た速度と指令を並べる。

両立版 (env/unified.py) は転倒しても終了しないので、「転んだか」と
「最後に立っていたか」を分けて出す。転んでも起き上がって歩き直した個体は
`転倒 あり / 最後は正立` になる。

使い方:
  python scripts/cmd_test.py --ckpt out/train/u_dagger/ckpt/final.pkl
  python scripts/cmd_test.py --ckpt ... --difficulty 1.0    # 段差 40mm 地形で
"""
import argparse, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--envs", type=int, default=64, help="指令あたりの試行数")
ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--difficulty", type=float, default=0.0,
                help="0=平地, 1.0=学習と同じ段差地形")
ap.add_argument("--contacts-per-env", type=int, default=32)
ap.add_argument("--settle", type=float, default=1.0,
                help="最初の何秒を集計から外すか (歩き出すまでの過渡)")
args = ap.parse_args()

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")

from env import _compat; _compat.install()
import jax
import jax.numpy as jp

from env import constants as consts
from env.unified import PupperUnified, unified_config
from scripts.policy import load

# 学習時のコマンド範囲 (env/constants.py) の内側で、代表的な向きを並べる。
CMDS = [
    ("前進",       [0.20, 0.00, 0.00]),
    ("後退",       [-0.20, 0.00, 0.00]),
    ("左横歩き",   [0.00, 0.12, 0.00]),
    ("右横歩き",   [0.00, -0.12, 0.00]),
    ("左旋回",     [0.00, 0.00, 0.60]),
    ("右旋回",     [0.00, 0.00, -0.60]),
    ("前進+旋回",  [0.15, 0.00, 0.40]),
    ("斜め前",     [0.15, 0.10, 0.00]),
    ("静止",       [0.00, 0.00, 0.00]),
]


def main():
    B = args.envs
    cfg = unified_config()
    cfg.recover.p_walk = 1.0              # 立った状態から始める (歩行だけを測る)
    cfg.terrain.difficulty = args.difficulty
    cfg.naconmax = args.contacts_per_env * B
    T = int(round(args.secs / cfg.ctrl_dt))
    cfg.episode_length = T
    env = PupperUnified(config=cfg)
    env._mjx_model = env._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
    _, inference, _ = load(args.ckpt, env=env)

    mm = env.mj_model
    UP = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id] + 2
    LIN = mm.sensor_adr[mm.sensor(consts.LOCAL_LINVEL_SENSOR).id]
    GYR = mm.sensor_adr[mm.sensor(consts.GYRO_SENSOR).id]

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    k0 = int(args.settle / cfg.ctrl_dt)

    print(f"[eval] {args.ckpt}")
    print(f"[env ] {'平地' if args.difficulty <= 0 else f'段差地形 difficulty={args.difficulty}'}"
          f"  {B} 試行/指令  {args.secs:.0f}s  最初の {args.settle:.0f}s は集計から除く\n")
    print("  指令            vx 指令/実測      vy 指令/実測      wz 指令/実測   "
          "  転倒  最後は正立")
    print("  " + "-" * 82)

    for name, c in CMDS:
        cmd = jp.array(c, jp.float32)
        st = reset(jax.random.split(jax.random.PRNGKey(0), B))
        v = np.zeros((T, B, 3))
        ups = np.zeros((T, B))
        for k in range(T):
            st.info["command"] = jp.tile(cmd, (B, 1))
            act, _ = inference(st.obs, jax.random.PRNGKey(k))
            st = step(st, act)
            d = st.data
            v[k, :, :2] = np.asarray(d.sensordata[:, LIN:LIN + 2])
            v[k, :, 2] = np.asarray(d.sensordata[:, GYR + 2])
            ups[k] = np.asarray(d.sensordata[:, UP])
        w = v[k0:]
        fell = (ups[k0:] < 0.5).any(0)          # 一度でも大きく傾いたか
        up_end = ups[-2] > 0.85                 # 最後に立っているか
        mv = w.mean(axis=(0, 1))
        print(f"  {name:<10s} {c[0]:+5.2f}/{mv[0]:+5.2f} m/s  "
              f"{c[1]:+5.2f}/{mv[1]:+5.2f} m/s  "
              f"{c[2]:+5.2f}/{mv[2]:+5.2f} rad/s   "
              f"{fell.mean()*100:4.0f}%  {up_end.mean()*100:5.0f}%")

    print("\n  vx/vy は機体座標の並進速度、wz はヨー角速度。"
          "転倒 = 一度でも水平度 0.5 未満 (両立版は転んでも終了しないので、"
          "起き上がれば「最後は正立」に戻る)")


if __name__ == "__main__":
    main()
