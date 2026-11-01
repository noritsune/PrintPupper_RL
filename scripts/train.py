"""PrintPupper 歩行ポリシーの PPO 学習 (brax / MJX)。

学習中の進捗が目で見えるように、一定間隔で
  - out/train/video/iter_XXXX.mp4   ロールアウト動画
  - out/train/progress.png          学習曲線
  - out/train/tb/                   TensorBoard ログ
  - out/train/ckpt/                 チェックポイント
を自動で書き出す。
"""
import argparse, functools, gc, json, os, pickle, sys, time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--num-timesteps", type=int, default=100_000_000)
ap.add_argument("--num-envs", type=int, default=4096)
ap.add_argument("--num-evals", type=int, default=25)
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--unroll-length", type=int, default=20)
ap.add_argument("--num-minibatches", type=int, default=32)
ap.add_argument("--updates-per-batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--entropy", type=float, default=0.005)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--impl", default="warp", choices=["warp", "jax"])
ap.add_argument("--no-randomize", action="store_true")
ap.add_argument("--contacts-per-env", type=int, default=8)
ap.add_argument("--num-eval-envs", type=int, default=128,
                help="評価に使う env 数。接触バッファの確保量に直結する")
ap.add_argument("--video-every", type=int, default=1, help="何回の eval ごとに動画を出すか")
ap.add_argument("--video-secs", type=float, default=6.0)
ap.add_argument("--out", default=os.path.join(ROOT, "out", "train"))
ap.add_argument("--tag", default=None)
ap.add_argument("--env", default="flat",
                choices=["flat", "terrain", "balance", "recover", "unified",
                         "climb"],
                help="flat=平地 (env/joystick.py), terrain=段差 (env/terrain.py), "
                     "balance=対角2脚での静止保持 (env/balance.py), "
                     "recover=転倒復帰 (env/recover.py), "
                     "unified=段差歩行+転倒復帰の両立 (env/unified.py), "
                     "climb=段差乗り越え (準静的登坂, env/climb.py)")
ap.add_argument("--discounting", type=float, default=None,
                help="既定: flat/terrain=0.97, balance/recover=0.99。balance は "
                     "「いま踏ん張ると1秒後に転ばない」を学ぶ必要があり、"
                     "0.97 (地平線 0.67s) では届かない。recover も「いま脚を"
                     "振ると1秒後に転がれる」が学習対象なので同じ")
ap.add_argument("--pert-kick", type=float, default=None,
                help="外乱キックの最大速度 [m/s]。balance は既定で外乱なし。"
                     "保持を覚えたあと、この値を入れて fine-tune する")
ap.add_argument("--scale", action="append", default=[], metavar="KEY=VALUE",
                help="報酬の重みを上書きする (例: --scale drift=-20)。複数指定可。"
                     "env のファイルを触らずに重みだけ振れる")
ap.add_argument("--spawn-radius", type=float, default=None, metavar="M",
                help="リセット時の水平位置のばらつき半径 [m] (既定 "
                     "constants.TERRAIN_SPAWN_R=3.0)。既定では中央±2mが平地なので "
                     "スポーン範囲のうち段差に遭遇する面積は 47%% しかなく、地形全体の "
                     "76%% より易しい。上げると1エピソードあたりの段差の練習量が増える")
ap.add_argument("--max-foot-height", type=float, default=None, metavar="M",
                help="遊脚の足上げ目標 [m]。既定は terrain_config が "
                     "SWING_HEIGHT + step_height で決めるが、段差を上げるとこれも "
                     "一緒に上がってしまう。律速が足上げでなく重心移送に移った後は "
                     "目標を上げると逆効果になる (run d40: 目標95mmで足は高く上がったが "
                     "越えられる段差は 40->30mm に悪化)。ここで据え置ける")
ap.add_argument("--gait-freq", type=float, nargs=2, default=None,
                metavar=("LOW", "HIGH"),
                help="位相クロックの周波数範囲 [Hz] (既定 constants.GAIT_FREQ)。"
                     "下げると遊脚時間が伸び、足を高く上げられる。実測: 遊脚の"
                     "上昇に使える時間 100ms で 49mm / 150ms で 81mm / 200ms で 88mm")
ap.add_argument("--episode-length", type=int, default=None,
                help="1エピソードの最大ステップ数 (既定は env の設定値)")
ap.add_argument("--difficulty", type=float, default=1.0,
                help="--env terrain のとき。0=完全な平地, 1=--step-height をフルに使う")
ap.add_argument("--step-height", type=float, default=None,
                help="--env terrain のとき。difficulty=1 での段差高さ [m]"
                     " (既定 constants.STEP_HEIGHT)")
ap.add_argument("--restore", default=None,
                help="params の .pkl を初期値にして fine-tune する。"
                     "観測次元が同じなら平地の重みをそのまま使える")
ap.add_argument("--action-scale", type=float, default=None, metavar="RAD",
                help="関節指令の残差スケール [rad] (既定: 歩行 0.25 / 復帰 1.6)。"
                     "方策出力は tanh で [-1,1] なので、可動域は厳密に "
                     "「基準姿勢 ± この値」。復帰は脚を大きく開かないと足が床に "
                     "届かないので 1.6 (全可動域) を使う")
ap.add_argument("--walk-action-scale", type=float, default=None, metavar="RAD",
                help="--env unified のとき、歩行側 (正立時) の残差スケール "
                     "cfg.unified.act_scale_walk [rad] を上書きする (既定 0.25)。"
                     "--action-scale は cfg.action_scale を書き換えるだけなので "
                     "unified の歩行側には効かない。足先の到達範囲はこの残差"
                     "スケールで決まる。実測 (股関節基準、最大足上げ / 足を "
                     "40mm 上げたときの前方到達): 0.25 -> 34.2mm / 到達不能、"
                     "0.35 -> 52.7mm / 71.9mm、0.45 -> 72.5mm / 82.7mm、"
                     "0.60 -> 102.9mm / 99.2mm。60mm の段に後脚を乗せるには"
                     "胴体が段の縁を +37mm 以上越えている必要があり、0.25 では"
                     "幾何的に届かない。上げると 1 目盛あたりが粗くなる代償が"
                     "ある (0.25 -> 0.60 で 2.4 倍)。**env/deploy.py の "
                     "ACTION_SCALE は 0.25 のままなので、ここを変えた重みを"
                     "実機に出すときは deploy.py 側も直すこと**。直さないと "
                     "scripts/check_deploy.py が「両立版の可動域の切り替えが"
                     "一致」で NG を出して検出する")
ap.add_argument("--obs-history", type=int, default=0, metavar="L",
                help="actor に観測履歴スタックを持たせる (既定 0 = 無効、38 次元)。"
                     "actor は IMU と行動履歴だけの盲目方策で、既定の記憶は "
                     "last_act + last_last_act の 2 ステップ = 40ms しかない。"
                     "遊脚は gait_freq 0.7-1.2Hz で 333-571ms なので 1 歩の "
                     "7-12%% しか見えていない。L フレーム分の過去の "
                     "[gyro, accel, gravity, action] (1 フレーム 21 次元) を "
                     "--obs-stride ステップおきに連結して記憶を持たせる "
                     "(観測は 38 + L*21 次元になる)。実効の時間窓は "
                     "L*stride*ctrl_dt 秒 (既定の 10*4*0.02 = 0.8s は歩容 "
                     "1 周期にほぼ相当)。**38 次元で学習済みの重みを引き継ぐには "
                     "scripts/adapt_ckpt.py で --obs-history を同じ値にして"
                     "変換してから --restore に渡すこと** (そのまま渡すと"
                     "次元不一致で落ちる)")
ap.add_argument("--obs-stride", type=int, default=4, metavar="S",
                help="--obs-history の間引き幅 [step] (既定 4)。生の L フレームを"
                     "全部入れると隣接フレームがほぼ同一値で冗長なので S "
                     "ステップおきに拾う。時間窓 = L*S*ctrl_dt 秒 なので、"
                     "L を増やさずに窓を伸ばしたいときはここを上げる")
ap.add_argument("--rec", action="append", default=[], metavar="KEY=VALUE",
                help="--env recover のとき cfg.recover.* を上書きする "
                     "(例: --rec p_stuck=0.2 --rec trunk_gate=False)。"
                     "初期姿勢の内訳や判定閾値を env のファイルを触らずに振れる")
ap.add_argument("--stuck-states", default=None, metavar="NPZ",
                help="--env recover のとき、前の run が実際に詰まった状態を "
                     "初期姿勢に混ぜる (scripts/make_stuck_states.py が作る)。"
                     "--rec p_stuck=0.2 と併せて使う")
ap.add_argument("--fresh-critic", action="store_true",
                help="--restore 時に critic だけ初期化する。報酬を変えた直後に有効")
args = ap.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45")

import jax
import jax.numpy as jp
import mujoco

from env import _compat
if _compat.install():
    print("[compat] jax.device_put_replicated をシムで復元 (brax 0.14.2 用)")

import warp as wp
wp.config.quiet = True
wp.config.verbose = False

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import mjx
from mujoco_playground._src import wrapper

from env.balance import PupperBalance, balance_config
from env.climb import PupperClimb, climb_config
from env.joystick import PupperJoystick, default_config
from env.recover import PupperRecover, recover_config
from env.terrain import PupperTerrain, terrain_config
from env.unified import PupperUnified, unified_config
from env.randomize import domain_randomize

RUN = args.tag or datetime.now().strftime("%m%d_%H%M")
OUT = os.path.join(args.out, RUN)
for sub in ("video", "ckpt", "tb"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
print(f"[run] {OUT}")
with open(os.path.join(OUT, "cmd.txt"), "w") as f:
    f.write(" ".join(sys.argv) + "\n")

if args.env == "terrain":
    from env import constants as consts
    step_h = consts.STEP_HEIGHT if args.step_height is None else args.step_height
    cfg = terrain_config(step_height=step_h)
    cfg.terrain.difficulty = args.difficulty
    env_cls = PupperTerrain
elif args.env == "balance":
    cfg = balance_config()
    env_cls = PupperBalance
elif args.env == "climb":
    # 登坂の既定の狙いは 60mm。帯は 30-70mm が置かれる (env/climb.py)。
    step_h = 0.060 if args.step_height is None else args.step_height
    cfg = climb_config(step_height=step_h)
    env_cls = PupperClimb
elif args.env == "recover":
    cfg = recover_config()
    env_cls = PupperRecover
elif args.env == "unified":
    from env import constants as consts
    # 両立版の既定の段差は 40mm (e40 と同じ)。terrain 版の既定 (30mm) ではない。
    step_h = 0.040 if args.step_height is None else args.step_height
    cfg = unified_config(step_height=step_h)
    cfg.terrain.difficulty = args.difficulty
    env_cls = PupperUnified
else:
    cfg = default_config()
    env_cls = PupperJoystick
for kv in args.scale:
    k, v = kv.split("=", 1)
    if k not in cfg.reward_config.scales:
        sys.exit(f"報酬項 '{k}' は {args.env} に無い。"
                 f"ある項目: {sorted(cfg.reward_config.scales.keys())}")
    print(f"[scale] {k}: {cfg.reward_config.scales[k]} -> {float(v)}")
    cfg.reward_config.scales[k] = float(v)
for kv in args.rec:
    k, v = kv.split("=", 1)
    if args.env not in ("recover", "unified"):
        sys.exit("--rec は --env recover / unified のときだけ使える")
    if k not in cfg.recover:
        sys.exit(f"cfg.recover に '{k}' は無い。ある項目: {sorted(cfg.recover.keys())}")
    cur = cfg.recover[k]
    new_v = (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v)
    print(f"[rec] {k}: {cur} -> {new_v}")
    cfg.recover[k] = new_v
if args.episode_length is not None:
    cfg.episode_length = args.episode_length
if args.pert_kick is not None:
    cfg.pert_config.enable = args.pert_kick > 0
    cfg.pert_config.velocity_kick = [0.0, args.pert_kick]
cfg.impl = args.impl
if args.action_scale is not None:
    cfg.action_scale = args.action_scale
if args.walk_action_scale is not None:
    if args.env != "unified":
        sys.exit("--walk-action-scale は --env unified のときだけ使える "
                 "(cfg.unified.act_scale_walk を書き換えるフラグ)。"
                 "他の env は --action-scale を使うこと")
    print(f"[act] unified 歩行側の可動域: {cfg.unified.act_scale_walk} -> "
          f"{args.walk_action_scale} rad (転倒側 {cfg.unified.act_scale_fallen} は据え置き)")
    cfg.unified.act_scale_walk = args.walk_action_scale
# 履歴スタックは全 env 共通 (default_config が持つ)。0 なら従来どおり 38 次元。
cfg.obs_history.length = args.obs_history
cfg.obs_history.stride = args.obs_stride
if args.spawn_radius is not None:
    if "terrain" not in cfg:
        sys.exit("--spawn-radius は --env terrain / unified のときだけ使える")
    cfg.terrain.spawn_radius = args.spawn_radius
if args.max_foot_height is not None:
    cfg.reward_config.max_foot_height = args.max_foot_height
if args.gait_freq is not None:
    cfg.gait_freq = list(args.gait_freq)
cfg.naconmax = args.contacts_per_env * args.num_envs
env = env_cls(config=cfg)

# 評価環境は num_eval_envs 個しか回さないので、接触バッファを学習env と同じ
# サイズで確保すると 32 倍の無駄になる。warp のバッファは XLA のプールの外で
# 取られるため、これが効いて eval の途中で確保が通らず落ちる (s30b が
# convex_narrowphase の EPA バッファ 71MB を確保できず OOM)。
# WSL では GPU の 1.5GB 程度が常時 Windows 側に取られており余裕が小さい。
eval_cfg = cfg.copy_and_resolve_references()
eval_cfg.naconmax = args.contacts_per_env * args.num_eval_envs
eval_env = env_cls(config=eval_cfg)
if args.stuck_states:
    n = env.load_stuck_states(args.stuck_states)
    eval_env.load_stuck_states(args.stuck_states)
    print(f"[stuck] {args.stuck_states} から {n} 状態を読み込み "
          f"(初期姿勢の {cfg.recover.p_stuck*100:.0f}% に混ぜる)")
    if cfg.recover.p_stuck <= 0:
        print("       (--rec p_stuck=... を指定しないと使われない)")
def _lift_mm(scale: float) -> float:
    """大腿・下腿の action_scale [rad] から足上げ上限 [mm] を引く (実測値の補間)。

    実測 (股関節基準): 0.25 -> 34.2mm / 0.35 -> 52.7mm / 0.45 -> 72.5mm /
    0.60 -> 102.9mm。表の外は端の値で頭打ちにする (np.interp の既定)。
    以前はここが 0.25 か否かの 2 択ベタ書きで、0.60 に上げても 72.5mm と
    表示していた。
    """
    return float(np.interp(scale, [0.25, 0.35, 0.45, 0.60],
                           [34.2, 52.7, 72.5, 102.9]))


if args.env == "terrain":
    tc = cfg.terrain
    print(f"[env] terrain difficulty={tc.difficulty} step={tc.step_height*1000:.0f}mm "
          f"実際の最大地形高さ={env._t_max*1000:.0f}mm "
          f"足上げ目標={cfg.reward_config.max_foot_height*1000:.0f}mm "
          f"phase_sigma={cfg.reward_config.feet_phase_sigma:.4f}")
elif args.env == "recover":
    rc = cfg.recover
    print(f"[env] recover 転倒復帰  action_scale={cfg.action_scale} rad "
          f"(可動域 = 基準姿勢 ± これ)  初期姿勢 仰向け{rc.p_flip*100:.0f}%/"
          f"横倒し{rc.p_side*100:.0f}%/正立{(1-rc.p_flip-rc.p_side)*100:.0f}%  "
          f"1エピソード {cfg.episode_length*cfg.ctrl_dt:.0f}s (途中終了なし)  "
          f"XML={PupperRecover.XML.name}")
elif args.env == "unified":
    tc, rc, uc = cfg.terrain, cfg.recover, cfg.unified
    print(f"[env] unified 段差歩行+転倒復帰  段差={tc.step_height*1000:.0f}mm "
          f"(difficulty={tc.difficulty}, 実際の最大地形高さ={env._t_max*1000:.0f}mm) "
          f"足上げ目標={cfg.reward_config.max_foot_height*1000:.0f}mm\n"
          f"      可動域 = 基準姿勢 ± {uc.act_scale_walk}rad (正立) .. "
          f"± {uc.act_scale_fallen}rad (転倒)、up {uc.gate_lo}-{uc.gate_hi} で補間\n"
          f"      初期姿勢 歩行{rc.p_walk*100:.0f}% / 転倒{(1-rc.p_walk)*100:.0f}% "
          f"(仰向け{rc.p_flip*100:.0f}%/横倒し{rc.p_side*100:.0f}%/"
          f"伏せ{rc.p_prone*100:.0f}%)  "
          f"1エピソード {cfg.episode_length*cfg.ctrl_dt:.0f}s (転倒では終了しない)  "
          f"XML={PupperUnified.XML.name}")
elif args.env == "climb":
    cc = cfg.climb
    print(f"[env] climb 段差乗り越え  狙い={args.step_height or 0.06:.3f}m "
          f"帯={[round(h*1000) for h in cc.heights]}mm\n"
          f"      可動域 (関節別) hip/upper/lower={list(cc.action_scale)} rad "
          f"-> 足上げ上限 {_lift_mm(cc.action_scale[1]):.0f}mm 相当 "
          f"(4節リンクの上限 183.7mm)\n"
          f"      歩容クロック停止 (gait_freq={list(cfg.gait_freq)}) / "
          f"静的安定余裕の目標={cc.support_target*1000:.0f}mm / "
          f"1エピソード {cfg.episode_length*cfg.ctrl_dt:.0f}s "
          f"(胴体接地では終了しない)")
elif args.env == "balance":
    bc = cfg.balance
    print(f"[env] balance 対角2脚保持  遊脚持ち上げ={list(bc.lift)} rad "
          f"目標クリアランス={bc.clearance_target*1000:.0f}mm "
          f"最大保持={cfg.episode_length*cfg.ctrl_dt:.1f}s "
          f"外乱={'あり ' + str(list(cfg.pert_config.velocity_kick)) if cfg.pert_config.enable else 'なし'}")
else:
    print("[env] flat")
if args.impl == "warp":                    # ソルバ警告のGPU printf が致命的に遅い
    for e in (env, eval_env):
        e._mjx_model = e._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})

BALANCE = args.env == "balance"
RECOVER = args.env == "recover"
UNIFIED = args.env == "unified"
CLIMB = args.env == "climb"

from env import constants as _consts
consts_stand = _consts.STAND_HEIGHT

# ---- ログ ----
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None
writer = SummaryWriter(os.path.join(OUT, "tb")) if SummaryWriter else None
history = []
t_start = time.time()


def progress(step, metrics):
    el = time.time() - t_start
    r = float(metrics.get("eval/episode_reward", np.nan))
    rs = float(metrics.get("eval/episode_reward_std", np.nan))
    ln = float(metrics.get("eval/avg_episode_length", np.nan))
    sps = step / max(el, 1e-9)
    history.append(dict(step=int(step), reward=r, reward_std=rs, ep_len=ln,
                        elapsed=el, sps=sps,
                        **{k: float(v) for k, v in metrics.items()
                           if k.startswith("eval/episode_reward/")}))
    # 報酬だけ見ていると「立ち止まって追従報酬を稼ぐ」失敗を見逃す。
    # 実速度と指令速度を必ず並べて出す。
    sp = float(metrics.get("eval/episode_track/speed", np.nan)) / max(ln, 1)
    csp = float(metrics.get("eval/episode_track/cmd_speed", np.nan)) / max(ln, 1)
    er = float(metrics.get("eval/episode_track/lin_err", np.nan)) / max(ln, 1)
    history[-1].update(speed=sp, cmd_speed=csp, lin_err=er)
    sat = float(metrics.get("eval/episode_hw/tau_sat", np.nan)) / max(ln, 1) * 100
    fl = float(metrics.get("eval/episode_hw/flight", np.nan)) / max(ln, 1) * 100
    du = float(metrics.get("eval/episode_hw/duty", np.nan)) / max(ln, 1)
    history[-1].update(tau_sat=sat, flight=fl, duty=du)
    # 遊脚の平均最高到達点。段差課題の成否はこれで決まるので常に出す
    # (報酬合計や速度だけ見ていると、足が上がっていないことに気付けない)。
    swp = float(metrics.get("eval/episode_swing_peak", np.nan)) / max(ln, 1) * 1000
    history[-1].update(swing_peak=swp)
    if UNIFIED:
        # 両立版は歩行と復帰の両方を 1 行で出す。片方だけ見ていると、
        # 「転ばないが段を避ける」「復帰はするが歩かない」を見逃す。
        suc = float(metrics.get("eval/episode_rec/success", np.nan)) / max(ln, 1)
        upf = float(metrics.get("eval/episode_rec/upright_frac", np.nan)) / max(ln, 1)
        trk = float(metrics.get("eval/episode_rec/trunk", np.nan)) / max(ln, 1)
        ft = float(metrics.get("eval/episode_rec/feet", np.nan)) / max(ln, 1)
        history[-1].update(success=suc, upright_frac=upf, trunk=trk, feet=ft)
        print(f"  [{el/60:6.1f}min] step {step:>11,}  r {r:7.2f}  ep {ln:6.1f}  "
              f"| 正立 {upf*100:4.0f}% 復帰 {suc*100:4.0f}% 胴体接地 {trk*100:3.0f}% "
              f"接地脚 {ft:.1f} "
              f"| 速度 {sp:.3f}/{csp:.3f} 遊脚 {swp:4.1f}mm "
              f"| トルク飽和 {sat:4.1f}% | {sps/1e3:5.1f}k/s", flush=True)
    elif RECOVER:
        # 成績は「復帰できたか」と「復帰までの時間」。エピソード長は固定なので
        # ep_len は見ても意味がない。rec/success は「一度でも 0.5s 連続で復帰
        # 条件を満たした」フラグの時間平均なので、早く復帰するほど 1 に近い
        # (復帰時刻 t なら約 (T-t)/T)。厳密な成功率は scripts/recover_test.py。
        upr = float(metrics.get("eval/episode_rec/upright", np.nan)) / max(ln, 1)
        okf = float(metrics.get("eval/episode_rec/ok", np.nan)) / max(ln, 1)
        suc = float(metrics.get("eval/episode_rec/success", np.nan)) / max(ln, 1)
        hgt = float(metrics.get("eval/episode_rec/height", np.nan)) / max(ln, 1)
        ft = float(metrics.get("eval/episode_rec/feet", np.nan)) / max(ln, 1)
        trk = float(metrics.get("eval/episode_rec/trunk", np.nan)) / max(ln, 1)
        history[-1].update(upright=upr, ok=okf, success=suc, height=hgt, feet=ft,
                           trunk=trk)
        # 復帰時刻の目安。suc は復帰後ずっと 1 なので T*(1-suc) が復帰時刻。
        t_rec = cfg.episode_length * cfg.ctrl_dt * (1.0 - suc)
        history[-1].update(t_rec=t_rec)
        print(f"  [{el/60:6.1f}min] step {step:>11,}  r {r:7.2f}  "
              f"| 復帰 {suc*100:4.0f}% (目安 {t_rec:4.1f}s) 条件成立 {okf*100:4.0f}% "
              f"| 水平度 {upr:+.2f} 胴体高 {hgt*1000:4.0f}mm 接地脚 {ft:.1f} "
              f"胴体接地 {trk*100:3.0f}% "
              f"| トルク飽和 {sat:4.1f}% | {sps/1e3:5.1f}k/s", flush=True)
    elif CLIMB:
        # 成績は「段の上に乗れたか」。報酬合計やエピソード長では読めない
        # (段の手前で立っているだけでも survival と support は稼げる)。
        suc = float(metrics.get("eval/episode_clb/success", np.nan)) / max(ln, 1)
        ft = float(metrics.get("eval/episode_clb/feet", np.nan)) / max(ln, 1)
        mg = float(metrics.get("eval/episode_clb/margin", np.nan)) / max(ln, 1)
        xx = float(metrics.get("eval/episode_clb/x", np.nan)) / max(ln, 1)
        hh = float(metrics.get("eval/episode_clb/height", np.nan)) / max(ln, 1)
        sh = float(metrics.get("eval/episode_clb/step_h", np.nan)) / max(ln, 1)
        history[-1].update(success=suc, feet=ft, margin=mg, x=xx, height=hh,
                           step_h=sh)
        on = float(metrics.get("eval/episode_clb/on_start", np.nan)) / max(ln, 1)
        fon = float(metrics.get("eval/episode_clb/feet_on", np.nan)) / max(ln, 1)
        son = float(metrics.get("eval/episode_clb/succ_on", np.nan)) / max(ln, 1)
        # 段上開始グループだけの群平均 (全体平均では2群が混ざって読めない)
        ft_on = fon / max(on, 1e-6)
        suc_on = son / max(on, 1e-6)
        xon = float(metrics.get("eval/episode_clb/x_on", np.nan)) / max(ln, 1)
        xwk = float(metrics.get("eval/episode_clb/x_walk", np.nan)) / max(ln, 1)
        x_on = xon / max(on, 1e-6)
        x_walk = xwk / max(1.0 - on, 1e-6)
        history[-1].update(feet_on=ft_on, success_on=suc_on,
                           x_on=x_on, x_walk=x_walk)
        print(f"  [{el/60:6.1f}min] step {step:>11,}  r {r:7.2f}  ep {ln:6.1f}  "
              f"| 登坂 {suc*100:4.0f}% 段上の足 {ft:.1f}/4 "
              f"[段上開始 {ft_on:.1f}/4 x{x_on*1000:+4.0f} 登坂 {suc_on*100:3.0f}% "
              f"| 手前開始 x{x_walk*1000:+5.0f}] "
              f"到達 x {xx*1000:+5.0f}mm 地面 {hh*1000:4.1f}/{sh*1000:4.1f}mm "
              f"| 安定余裕 {mg*1000:+5.1f}mm | トルク飽和 {sat:4.1f}% "
              f"| {sps/1e3:5.1f}k/s", flush=True)
    elif BALANCE:
        # この課題の成績はエピソード長そのもの (遊脚が触れたら即終了)。
        hold = ln * cfg.ctrl_dt
        sw = float(metrics.get("eval/episode_bal/swing_z", np.nan)) / max(ln, 1) * 1000
        ti = float(metrics.get("eval/episode_bal/tilt", np.nan)) / max(ln, 1)
        stc = float(metrics.get("eval/episode_bal/stance", np.nan)) / max(ln, 1) * 100
        history[-1].update(hold=hold, swing_z=sw, tilt=ti, stance=stc)
        print(f"  [{el/60:6.1f}min] step {step:>11,}  r {r:7.2f}  "
              f"| 保持 {hold:5.2f}s / {cfg.episode_length*cfg.ctrl_dt:.0f}s "
              f"| 遊脚 {sw:4.1f}mm 傾き {ti:.3f} 接地脚 {stc:4.0f}% "
              f"ずれ {er*1000:3.0f}mm | トルク飽和 {sat:4.1f}% "
              f"| {sps/1e3:5.1f}k/s", flush=True)
    else:
        print(f"  [{el/60:6.1f}min] step {step:>11,}  r {r:7.2f}  ep {ln:6.1f}  "
              f"| 速度 {sp:.3f}/{csp:.3f} 誤差 {er:.3f} "
              f"| 遊脚 {swp:4.1f}mm "
              f"| トルク飽和 {sat:4.1f}% 空中 {fl:4.1f}% duty {du:.2f} "
              f"| {sps/1e3:5.1f}k/s", flush=True)
    if writer:
        for k, v in metrics.items():
            try: writer.add_scalar(k, float(v), step)
            except Exception: pass
        writer.add_scalar("perf/steps_per_sec", sps, step)
        writer.flush()
    with open(os.path.join(OUT, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    _plot()


def _plot():
    if len(history) < 2:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = history
    st = [x["step"] for x in h]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    r = np.array([x["reward"] for x in h]); s = np.array([x["reward_std"] for x in h])
    ax[0].plot(st, r, lw=2); ax[0].fill_between(st, r - s, r + s, alpha=0.25)
    ax[0].set_title("episode reward"); ax[0].set_xlabel("env steps")
    if UNIFIED:
        ax[1].plot(st, [x.get("upright_frac", np.nan) * 100 for x in h], lw=2,
                   color="tab:green", label="upright [% of episode]")
        ax[1].plot(st, [x.get("success", np.nan) * 100 for x in h], lw=1.5,
                   color="tab:orange", label="recovered (time-avg) [%]")
        ax[1].set_title("upright / recovery"); ax[1].set_xlabel("env steps")
        ax[1].set_ylim(-2, 102); ax[1].legend(fontsize=8)
        ax[2].plot(st, [x.get("speed", np.nan) for x in h], lw=2, label="speed [m/s]")
        ax[2].plot(st, [x.get("cmd_speed", np.nan) for x in h], lw=1.5,
                   label="cmd speed [m/s]")
        ax[2].plot(st, [x.get("swing_peak", np.nan) / 1000 for x in h], lw=1.5,
                   label="swing peak [m]")
        ax[2].set_title("walking"); ax[2].set_xlabel("env steps")
        ax[2].legend(fontsize=8)
    elif RECOVER:
        ax[1].plot(st, [x.get("success", np.nan) * 100 for x in h], lw=2,
                   color="tab:orange", label="recovered (time-avg) [%]")
        ax[1].plot(st, [x.get("ok", np.nan) * 100 for x in h], lw=1.5,
                   color="tab:blue", label="recovery condition met [%]")
        ax[1].set_title("recovery"); ax[1].set_xlabel("env steps")
        ax[1].set_ylim(-2, 102); ax[1].legend(fontsize=8)
        ax[2].plot(st, [x.get("upright", np.nan) for x in h], lw=2,
                   label="uprightness (-1 flipped .. +1 upright)")
        ax[2].plot(st, [x.get("height", np.nan) * 10 for x in h], lw=1.5,
                   color="tab:green", label="base height x10 [m]")
        ax[2].plot(st, [x.get("feet", np.nan) / 4 for x in h], lw=1.5,
                   color="tab:red", label="feet on ground / 4")
        ax[2].plot(st, [x.get("trunk", np.nan) for x in h], lw=1.5,
                   color="tab:purple", ls="--", label="trunk touching floor")
        ax[2].axhline(consts_stand * 10, ls=":", color="gray")
        ax[2].set_title("posture"); ax[2].set_xlabel("env steps")
        ax[2].legend(fontsize=8)
    elif CLIMB:
        ax[1].plot(st, [x.get("success", np.nan) * 100 for x in h], lw=2,
                   color="tab:orange", label="climbed (time-avg) [%]")
        ax[1].plot(st, [x.get("feet", np.nan) / 4 * 100 for x in h], lw=1.5,
                   color="tab:blue", label="feet on ledge / 4 [%]")
        ax[1].set_title("climb"); ax[1].set_xlabel("env steps")
        ax[1].set_ylim(-2, 102); ax[1].legend(fontsize=8)
        ax[2].plot(st, [x.get("x", np.nan) * 1000 for x in h], lw=2,
                   label="trunk x from ledge [mm]")
        ax[2].plot(st, [x.get("height", np.nan) * 1000 for x in h], lw=1.5,
                   color="tab:green", label="ground under feet [mm]")
        ax[2].plot(st, [x.get("margin", np.nan) * 1000 for x in h], lw=1.5,
                   color="tab:red", label="support margin [mm]")
        ax[2].axhline(0, ls=":", color="gray")
        ax[2].set_title("progress"); ax[2].set_xlabel("env steps")
        ax[2].legend(fontsize=8)
    elif BALANCE:
        ax[1].plot(st, [x.get("hold", np.nan) for x in h], lw=2, color="tab:orange")
        ax[1].axhline(cfg.episode_length * cfg.ctrl_dt, ls=":", color="gray")
        ax[1].set_title("hold time [s] (diagonal 2-leg stand)")
        ax[1].set_xlabel("env steps")
        ax[2].plot(st, [x.get("swing_z", np.nan) for x in h], lw=2,
                   label="swing foot clearance [mm]")
        ax[2].plot(st, [x.get("tilt", np.nan) * 100 for x in h], lw=1.5,
                   color="tab:red", label="body tilt x100 [-]")
        ax[2].plot(st, [x.get("lin_err", np.nan) * 1000 for x in h], lw=1.5,
                   color="tab:green", label="drift [mm]")
        ax[2].axhline(cfg.balance.clearance_target * 1000, ls=":", color="gray")
        ax[2].set_title("posture"); ax[2].set_xlabel("env steps")
        ax[2].legend(fontsize=8)
    else:
        ax[1].plot(st, [x["ep_len"] for x in h], lw=2, color="tab:orange")
        ax[1].set_title("episode length (max 1000)"); ax[1].set_xlabel("env steps")
        ax[2].plot(st, [x.get("speed", np.nan) for x in h], lw=2, label="achieved |v|")
        ax[2].plot(st, [x.get("cmd_speed", np.nan) for x in h], lw=2, ls="--",
                   label="commanded |v|")
        ax[2].plot(st, [x.get("lin_err", np.nan) for x in h], lw=1.5, color="tab:red",
                   label="tracking error")
        ax[2].set_title("velocity tracking [m/s]"); ax[2].set_xlabel("env steps")
        ax[2].legend(fontsize=8)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "progress.png"), dpi=110)
    plt.close(fig)


# ---- 動画書き出し ----
_render = {"n": 0}


def render_rollout(make_policy, params, path, secs):
    """学習中のポリシーで 1 エピソード回して mp4 に落とす。"""
    import imageio
    inference = jax.jit(make_policy(params, deterministic=True))
    jit_reset, jit_step = jax.jit(eval_env.reset), jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(7)
    state = jit_reset(rng)
    # 見せたい指令: 前進 -> 旋回 -> 横歩き
    n = int(secs / eval_env.dt)
    seg = max(1, n // 3)
    cmds = [jp.array([0.18, 0.0, 0.0]), jp.array([0.10, 0.0, 0.6]),
            jp.array([0.0, 0.09, 0.0])]
    traj, used = [], []
    for k in range(n):
        # balance/recover では command 枠に速度指令を入れない。
        # 両立版は入れる (転倒 -> 復帰 -> その指令で歩き出す、までが見たい)。
        if not (BALANCE or RECOVER):
            state.info["command"] = cmds[min(k // seg, 2)]
        act, _ = inference(state.obs, jax.random.PRNGKey(k))
        state = jit_step(state, act)
        traj.append(state)
        used.append(np.array(state.info["command"]))
    # mujoco_playground の render_array (= eval_env.render) は内部で作った
    # mujoco.Renderer を閉じない。実測ではリークは 5 本で +3MiB 程度しかなく
    # OOM の原因ではなかったが (それは eval 環境の接触バッファ過大が原因)、
    # 明示的に閉じておく。
    mj = eval_env.mj_model
    d = mujoco.MjData(mj)
    frames = []
    with mujoco.Renderer(mj, height=360, width=480) as r:
        for st in traj[::2]:
            d.qpos[:] = np.array(st.data.qpos)
            d.qvel[:] = np.array(st.data.qvel)
            mujoco.mj_forward(mj, d)
            r.update_scene(d, camera="track")
            frames.append(r.render())
    imageio.mimsave(path, frames, fps=int(1 / eval_env.dt / 2), quality=8)
    del frames
    gc.collect()
    xy = np.array([np.array(s.data.qpos[:2]) for s in traj])
    alive = sum(float(s.done) == 0 for s in traj)
    if RECOVER or UNIFIED:
        ups = np.array([float(s.metrics["rec/upright"]) for s in traj])
        oks = np.array([float(s.metrics["rec/ok"]) for s in traj])
        first = next((i for i, v in enumerate(oks) if v > 0), None)
        print(f"    -> {os.path.basename(path)}  水平度 {ups[0]:+.2f} -> "
              f"{ups[-1]:+.2f} (最大 {ups.max():+.2f})  "
              f"復帰条件 初到達 "
              f"{'なし' if first is None else f'{first*eval_env.dt:.2f}s'}  "
              f"成立 {oks.mean()*100:.0f}% の時間", flush=True)
    elif BALANCE:
        # 途中で転んでも自動リセットはされない (eval_env は素の env)。
        # 最初に done が立つまでが保持時間。
        first = next((i for i, s in enumerate(traj) if float(s.done) > 0), n)
        print(f"    -> {os.path.basename(path)}  保持 {first*eval_env.dt:.2f}s "
              f"(動画 {secs:.0f}s ぶん), 移動 {np.linalg.norm(xy[-1]-xy[0]):.2f} m",
              flush=True)
    else:
        print(f"    -> {os.path.basename(path)}  移動 "
              f"{np.linalg.norm(xy[-1]-xy[0]):.2f} m, "
              f"生存 {alive}/{n} step", flush=True)


def policy_params_fn(current_step, make_policy, params):
    with open(os.path.join(OUT, "ckpt", f"params_{current_step:012d}.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(OUT, "ckpt", "latest.pkl"), "wb") as f:
        pickle.dump(params, f)
    _render["n"] += 1
    if args.video_every and _render["n"] % args.video_every == 0:
        try:
            render_rollout(make_policy, params,
                           os.path.join(OUT, "video", f"iter_{current_step:012d}.mp4"),
                           args.video_secs)
        except Exception as ex:
            print(f"    (動画書き出し失敗: {type(ex).__name__}: {ex})", flush=True)


network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 128, 64),
    value_hidden_layer_sizes=(256, 128, 64),
    policy_obs_key="state",           # actor: 実機で取れる観測のみ
    value_obs_key="privileged_state",  # critic: 特権情報あり (非対称 actor-critic)
)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=args.num_timesteps,
    num_evals=args.num_evals,
    episode_length=cfg.episode_length,
    num_envs=args.num_envs,
    num_eval_envs=args.num_eval_envs,
    batch_size=args.batch_size,
    unroll_length=args.unroll_length,
    num_minibatches=args.num_minibatches,
    num_updates_per_batch=args.updates_per_batch,
    learning_rate=args.lr,
    entropy_cost=args.entropy,
    discounting=(args.discounting if args.discounting is not None
                 else (0.99 if (BALANCE or RECOVER)
                       else 0.98 if UNIFIED else 0.97)),
    gae_lambda=0.95,
    clipping_epsilon=0.2,
    normalize_observations=True,
    action_repeat=1,
    reward_scaling=1.0,
    max_grad_norm=1.0,
    network_factory=network_factory,
    randomization_fn=None if args.no_randomize else domain_randomize,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    progress_fn=progress,
    policy_params_fn=policy_params_fn,
    seed=args.seed,
)

if args.restore:
    # brax の restore_params は (normalizer, policy, value) の 3 つ組を取る。
    # actor 観測が同じなら平地の重みをそのまま初期値にできる。報酬を変えた場合
    # critic の価値推定はずれているので --fresh-critic で捨てられるようにしてある。
    with open(args.restore, "rb") as f:
        restored = pickle.load(f)
    train_fn = functools.partial(train_fn, restore_params=restored,
                                 restore_value_fn=not args.fresh_critic)
    print(f"[restore] {args.restore}  critic="
          f"{'初期化' if args.fresh_critic else '引き継ぎ'}")

print(f"[cfg] envs={args.num_envs} timesteps={args.num_timesteps:,} impl={args.impl} "
      f"randomize={not args.no_randomize}")
_oh = cfg.obs_history
_hist = ("なし (38次元のまま)" if _oh.length == 0 else
         f"{_oh.length}frame x stride {_oh.stride} "
         f"= 時間窓 {_oh.length * _oh.stride * cfg.ctrl_dt:.2f}s")
print(f"[obs] actor={env.observation_size['state']} "
      f"critic={env.observation_size['privileged_state']} 履歴={_hist}")
make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
with open(os.path.join(OUT, "ckpt", "final.pkl"), "wb") as f:
    pickle.dump(params, f)
print(f"[done] {(time.time()-t_start)/60:.1f} min -> {OUT}")
