"""学習時の観測と実機デプロイ用の観測が完全に一致するかを突き合わせる。

ここがずれていると「sim では歩くのに実機では倒れる」の最もありふれた原因になる。
順序・スケール・位相の進め方まで含めて数値で比較する。
"""
import argparse, os, sys
import numpy as np, jax, jax.numpy as jp

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None,
                help="指定すると brax の推論と numpy 実装 (NumpyPolicy) の"
                     "行動が一致するかも検査する。実機・ビューアはこの numpy"
                     "実装を使うので、ここがずれると sim だけ動く方策になる")
ap.add_argument("--obs-history", type=int, default=10, metavar="L",
                help="観測履歴スタックのフレーム数 (0 で無効)。学習時の "
                     "cfg.obs_history.length と同じ値を入れる。先頭 38 次元は "
                     "履歴の有無で変わらないので、L>0 で回せば履歴なしの経路も"
                     "同時に検査できる")
ap.add_argument("--obs-stride", type=int, default=4, metavar="S",
                help="観測履歴スタックの間引き間隔 (学習時と同じ値)")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

from env.joystick import PupperJoystick, default_config
from env import constants as consts
from env.deploy import ObsBuilder, joint_to_servo, pwm_saturated, DEFAULT_POSE

cfg = default_config()
cfg.imu_config.level = 0.0        # ノイズを消して構造だけを比較する
# 姿勢推定の加速度混入も切る。実機ではフィルタ出力をそのまま ObsBuilder に渡すので、
# ここでは「同じ重力ベクトルを渡したら同じ観測になるか」だけを見たい。
cfg.imu_config.gravity_accel_coupling = 0.0
cfg.pert_config.enable = False
cfg.action_delay.enable = False
cfg.obs_history.length = args.obs_history
cfg.obs_history.stride = args.obs_stride
env = PupperJoystick(config=cfg)
reset, step = jax.jit(env.reset), jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))
gait_freq = float(state.info["gait_freq"])
ob = ObsBuilder(gait_freq=gait_freq, history_len=args.obs_history,
                history_stride=args.obs_stride)
ob.phase = float(state.info["phase"])
cmd = np.array([0.22, 0.05, -0.3], np.float32)

SLICES = [("gyro", 0, 3), ("accel", 3, 6), ("gravity", 6, 9), ("command", 9, 12),
          ("last_act", 12, 24), ("last_last_act", 24, 36), ("phase", 36, 38)]
if args.obs_history > 0:
    # 履歴部。リングバッファの詰め方・間引き位置・立ち上がり (電源投入直後の
    # ゼロ埋め) まで含めて 1 ステップ目から一致していなければならない。
    SLICES.append(("obs_history", 38, 38 + args.obs_history * 21))
rng = np.random.default_rng(0)
max_err = 0.0
per_slice = {n: 0.0 for n, _, _ in SLICES}
sat_total = 0
for k in range(300):
    state.info["command"] = jp.array(cmd)
    # 実機で読める量だけを取り出して deploy 側に渡す
    gyro = np.array(env._sensor(state.data, consts.GYRO_SENSOR))
    accel = np.array(env._sensor(state.data, consts.ACCELEROMETER_SENSOR))
    grav = np.array(env._gravity(state.data))
    o_deploy = ob.step(gyro, accel, grav, cmd)

    action = rng.uniform(-1, 1, 12).astype(np.float32)
    o_env = np.array(state.obs["state"])
    if k >= 1:                      # k=0 は reset 直後で初期条件がまだ揃わない
        max_err = max(max_err, np.abs(o_env - o_deploy).max())
        for n, a, b in SLICES:
            per_slice[n] = max(per_slice[n], np.abs(o_env[a:b] - o_deploy[a:b]).max())

    joint_cmd = ob.commit(action)
    sat_total += int(pwm_saturated(joint_to_servo(joint_cmd)).sum())
    state = step(state, jp.array(action))

print("=" * 66)
print("学習時の観測 vs 実機デプロイ用の観測")
print("=" * 66)
print(f"  観測次元 {env.observation_size['state'][0]} "
      f"(履歴 length={args.obs_history} stride={args.obs_stride} = "
      f"{args.obs_history*args.obs_stride*cfg.ctrl_dt:.2f}s の窓)")
print(f"  {'成分':<16s}{'最大絶対誤差':>14s}")
for n, _, _ in SLICES:
    print(f"  {n:<16s}{per_slice[n]:>14.3e}")
print(f"  {'合計':<16s}{max_err:>14.3e}")
print()
# 位相だけは内部クロックの float32 累積差が出る (実機は自前のクロックを使うので
# そもそも一致する必要がない)。それ以外は完全一致でなければならない。
sensor_err = max(v for n, v in per_slice.items() if n != "phase")
ok = sensor_err < 1e-6 and per_slice["phase"] < 1e-3
print(f"  {'[OK  ]' if ok else '[NG  ]'} センサ/コマンド/行動履歴が完全一致 "
      f"(誤差 {sensor_err:.1e})")
print(f"  {'[OK  ]' if per_slice['phase'] < 1e-3 else '[NG  ]'} 位相の差は float32 "
      f"累積ぶんのみ ({per_slice['phase']:.1e})")
print()
if args.ckpt:
    # 観測が合っていても、方策の再実装がずれていれば実機では別物が動く。
    # 隠れ層の活性化 (brax は silu、tanh ではない) を間違えた実績があるので
    # ここで必ず数値で突き合わせる。
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from policy import NumpyPolicy, load
    _, inference, _ = load(args.ckpt, env=env)
    npp = NumpyPolicy(args.ckpt)
    rng2 = np.random.default_rng(1)
    a_err = 0.0
    st = reset(jax.random.PRNGKey(0))
    for k in range(100):
        o = np.array(st.obs["state"])
        a_brax = np.array(inference(st.obs, jax.random.PRNGKey(k))[0])
        a_err = max(a_err, np.abs(a_brax - npp(o)).max())
        st = step(st, jp.array(a_brax))
    ok_a = a_err < 1e-4
    print(f"  {'[OK  ]' if ok_a else '[NG  ]'} brax の推論と numpy 実装の行動が"
          f"一致 (最大差 {a_err:.1e})")
    print()
    ok = ok and ok_a

# --- 両立版 (env/unified.py) の切り替え規則が実機側と一致するか ---
# ここがずれると「sim では起き上がるのに実機では可動域が開かない」という形で
# 静かに失敗する。学習側 (env/unified.py) と実機側 (env/deploy.py) で、
#   残差スケール / 速度コマンドを無視する条件 / 歩容位相を止める条件
# の 3 つが同じ重力ベクトルに対して同じ結果になることを数値で突き合わせる。
from env.deploy import unified_action_scale, upright_gate, GATE_OBS_SWITCH
from env.unified import PupperUnified, unified_config

ucfg = unified_config()
ucfg.naconmax = 24 * 8
uenv = PupperUnified(config=ucfg)
gate_err = scale_err = 0.0
for tilt in np.linspace(0.0, np.pi, 181):          # 正立 -> 仰向け
    g = np.array([np.sin(tilt), 0.0, -np.cos(tilt)], np.float32)   # 重力方向
    up = float(-g[2])
    gate_err = max(gate_err, abs(float(uenv._gate(jp.array(up))) - upright_gate(g)))
    scale_err = max(scale_err,
                    abs(float(uenv._act_scale(jp.array(up))) - unified_action_scale(g)))
# しきい値: 学習側は jax の float32、実機側は numpy の float32/float64 で
# smoothstep を計算するので、丸めの差が数 ppm 残る。可動域 1.6rad に対して
# 4e-6 rad は PWM 1 カウント (約 2e-4 rad) の 50 分の 1 で、意味を持たない。
# ここで見たいのは「式や帯の値がずれていないか」なので 1e-4 で切る。
ok_u = gate_err < 1e-4 and scale_err < 1e-4
print(f"  {'[OK  ]' if ok_u else '[NG  ]'} 両立版の可動域の切り替えが学習側と"
      f"実機側で一致 (ゲート {gate_err:.1e} / スケール {scale_err:.1e})")
print(f"         正立 ±{unified_action_scale(np.array([0,0,-1.0])):.2f}rad / "
      f"横倒し ±{unified_action_scale(np.array([1.0,0,0])):.2f}rad / "
      f"仰向け ±{unified_action_scale(np.array([0,0,1.0])):.2f}rad  "
      f"(コマンド/位相を止めるのは ゲート <= {GATE_OBS_SWITCH})")
ok = ok and ok_u
print()

print(f"  ランダム行動 300 ステップ中の PWM 飽和 = {sat_total} / {300*12} 関節ステップ"
      f"  ({sat_total/(300*12)*100:.1f}%)")
print("  (ランダム行動なので飽和が出るのは正常。学習後の方策で 0 に近いことを確認する)")
sys.exit(0 if ok else 1)
