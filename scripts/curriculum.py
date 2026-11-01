"""段差カリキュラム: e40 から 40 -> 50 -> 60mm へ段階的に fine-tune する。

## なぜ u_stand ではなく e40 から始めるのか (実測で分かったこと)

最初は実機に載っている u_stand (両立版) から始めようとしたが、**PPO を回した
瞬間に壊れた** (8M step で報酬 64.5 -> -47.8、正立 93% -> 56%、胴体接地 4% -> 41%)。
観測履歴を外しても学習率を下げても止まらなかった。

原因は u_stand が **PPO の学習結果ではなく蒸留 (scripts/distill.py) の産物**
だったこと。3 つの教師 (e40 の歩行 / u6 の復帰 / 静止は出力 0) を真似ただけなので、
**unified の報酬関数の最適点に乗っていない**。critic も対応していない。
PPO はそこから「報酬が実際に得な方向」へ即座に歩き出す。報酬の内訳がそれを示す:

    feet_clearance  -1540 -> -1015   <- 唯一改善した項
    fallen           -398 -> -2509
    upright          3700 -> 2333
    trunk_contact     -66 ->  -791

足上げ目標 85mm に対し実際の遊脚は 49mm なので feet_clearance は常に大赤字。
PPO はそこだけを取りに行き、遊脚を 49 -> 141mm に上げ、代償に転倒した。
OUTPUTS.md の run u4 で記録されている「報酬は上がるが挙動は悪化する」型そのもの。

**段差性能を伸ばす場所は地形歩行の本流 (d30b -> g30 -> e40)**。e40 は本物の
PPO チェックポイントなので、継続しても報酬の最適点から歩き出さない
(実測: 30M step 回して報酬 7.7 -> 8.0 で安定)。転倒復帰は段差登坂と無関係なので、
**最後に scripts/distill.py で u6 と合流させて実機用の両立版を作る**
(OUTPUTS.md「仕上がった復帰 (u6) と歩行 (e40) を蒸留し直す」と同じ手順)。

## なぜ専用スキルではなくカリキュラムなのか

OUTPUTS.md の「段差乗り越えを別スキルとして立てる」で、登坂専用の env/climb.py を
21 本回した結果は **60mm 成功率 0%** で、汎用方策 u_stand (40mm を 70.8%) に完敗した。
一方で 30mm -> 40mm を通した d30b -> e40 の「学習地形の段差を少しずつ上げる
fine-tune」は実際に効いている。**効いた手をもう一段回す**のがこのスクリプト。

同時に、OUTPUTS.md が「次の一手」に挙げて未着手のまま残っていた 3 つを全部入れる:

  1. 観測履歴スタック (--obs-history)
     actor は IMU と行動履歴しか持たず、既定の記憶は last_act + last_last_act の
     2 ステップ = 40ms しかない。遊脚は gait_freq 0.7-1.2Hz で 333-571ms なので
     **1 歩の 7-12% しか見えていない**。「さっき踏んだ足は床か段の上か」
     「胴体はもう段の縁を越えたか」は原理的に表現できず、位相クロックという
     open-loop の時計しか頼るものがない。0.8s の窓を与えて接地と重心の移送を
     推定させる。実機の追加ハードは要らない (IMU と自分の指令だけ)。

  2. 残差スケールの引き上げ (--action-scale)
     足先の到達範囲は残差スケールで決まる。実測 (股関節基準、scripts の
     env/linkage.py で再現できる):

       scale  最大足上げ  足を 40mm 上げたときの前方到達
       0.25     34.2mm      到達不能
       0.35     52.7mm      71.9mm
       0.45     72.5mm      82.7mm
       0.60    102.9mm      99.2mm

     60mm の段に後脚を乗せる条件は「胴体が段の縁を +37mm 以上越えてから」で、
     歩行の既定 0.25 では幾何的に届かない。**律速は足上げでも報酬でもなく、
     まずここ**。ただし上げると 1 目盛あたりが粗くなる (0.25 -> 0.35 で 1.4 倍)
     ので、必要な段 (50mm 以上) に入ってから上げる。

  3. 学習の延長
     c60t は 150M step で報酬が 51 -> 104 と単調増加したまま打ち切られていた。
     各段を 250-300M 取る。

## 重みの引き継ぎ (ここが肝心)

段の間で観測次元と残差スケールが変わるので、そのまま --restore すると落ちるか、
落ちなくても意味が変わる。scripts/adapt_ckpt.py が手術する:

  - 観測次元 38 -> 248: 入力重みの履歴部を **ゼロで埋める**ので、
    初期状態は元の方策と数値的に完全に等価。e40 の歩行能力を 1 ステップも
    失わずに、履歴の使い方だけを上乗せしていく。
  - 残差スケール A -> B: 最終層の行動平均側を A/B 倍する。tanh の前を縮めるだけ
    なので **近似的な暖機**であって等価変換ではない (飽和している出力はずれる)。

## 使い方

    python scripts/curriculum.py                 # 全段を順に回す
    python scripts/curriculum.py --dry-run       # コマンドだけ表示
    python scripts/curriculum.py --from-stage 2  # 途中から (前段の final.pkl を使う)

各段の後に scripts/step_test.py で「越えられる段差」を測って表に出す。
**報酬ではなくこの数字で判断すること** (OUTPUTS.md「報酬が上がっても目的が
動いていないことがある」)。
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(ROOT, "out/train/e40/ckpt/final.pkl"),
                help="出発点の重み。既定は段差歩行の本流の最終形 e40 "
                     "(40mm を 70.8%%、観測 38 次元、残差スケール 0.25)。"
                     "**蒸留産物 (u_stand) を渡してはいけない** — 報酬関数の"
                     "最適点に乗っていないので PPO が即座に壊す")
ap.add_argument("--prefix", default="cur", help="run タグの接頭辞")
ap.add_argument("--obs-history", type=int, default=10, metavar="L")
ap.add_argument("--obs-stride", type=int, default=4, metavar="S")
ap.add_argument("--from-stage", type=int, default=1, help="この段から再開 (1 始まり)")
ap.add_argument("--to-stage", type=int, default=None)
ap.add_argument("--num-envs", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-4,
                help="既定の 3e-4 ではなく 1e-4。観測履歴で入力が 38 -> 248 次元に"
                     "増えると hidden_0 に流れる勾配が入力数に比例して大きくなり、"
                     "収束済みの重みには 3e-4 は強すぎる。実測 (30M step, e40 から継続): "
                     "履歴なし 3e-4 で 7.7->8.0 安定 / 履歴あり 3e-4 で 7.5->3.0 悪化 / "
                     "履歴あり 1e-4 で 7.6->9.1 (ピーク 11.4) と対照を上回る")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--no-eval", action="store_true", help="段ごとの step_test を省く")
args = ap.parse_args()

# --- 各段の定義 ---
# step:  学習地形の段差 [m]
# scale: 関節指令の残差スケール [rad] (cfg.action_scale)
# steps: 学習ステップ数
#
# 第 1 段で段差を上げないのは意図的。**観測次元が変わった直後に難易度も上げると、
# どちらが効いた / 壊したのか切り分けられなくなる**。40mm は e40 が 70.8% で
# 越えられる高さなので、ここでの成績が e40 を下回ったら履歴の入れ方が
# 間違っている、と読める (d40 / s50 で「地形を上げたら悪化した」を踏んでいる)。
STAGES = [
    dict(tag="t40", step=0.040, scale=0.25, steps=150_000_000,
         note="観測履歴を入れて、e40 が既に越えられる 40mm で使い方を覚える"),
    dict(tag="t50", step=0.050, scale=0.35, steps=250_000_000,
         note="50mm。残差スケールを 0.25 -> 0.35 に上げて足先の到達を確保する"),
    dict(tag="t60", step=0.060, scale=0.35, steps=300_000_000,
         note="60mm。目標の高さ"),
]

lo = max(1, args.from_stage)
hi = args.to_stage or len(STAGES)


def run(cmd, what):
    print(f"\n{'='*72}\n[{what}] {' '.join(str(c) for c in cmd)}\n{'='*72}", flush=True)
    if args.dry_run:
        return 0
    return subprocess.call([str(c) for c in cmd])


def ckpt_of(tag):
    return os.path.join(ROOT, "out", "train", f"{args.prefix}_{tag}", "ckpt", "final.pkl")


manifest = []
for i, st in enumerate(STAGES, 1):
    if not (lo <= i <= hi):
        continue
    tag = f"{args.prefix}_{st['tag']}"
    prev = STAGES[i - 2] if i >= 2 else None
    src = args.base if i == 1 else ckpt_of(STAGES[i - 2]["tag"])
    # 前段の観測次元 / 残差スケール。第 1 段の出発点 (e40) だけ履歴なし 0.25。
    src_hist = 0 if i == 1 else args.obs_history
    src_scale = 0.25 if i == 1 else prev["scale"]

    print(f"\n\n#### 段 {i}/{len(STAGES)}: {tag} — {st['note']}")
    print(f"     段差 {st['step']*1000:.0f}mm / 残差スケール {st['scale']} rad "
          f"/ {st['steps']/1e6:.0f}M step")
    if not args.dry_run and not os.path.exists(src):
        sys.exit(f"[ERR] 出発点の重みが無い: {src}\n"
                 f"      前の段が終わっていないか、--from-stage が進みすぎている")

    # --- 重みの手術 (必要なときだけ) ---
    restore = src
    need_hist = src_hist != args.obs_history
    need_scale = abs(src_scale - st["scale"]) > 1e-9
    if need_hist or need_scale:
        restore = os.path.join(ROOT, "out", "train", f"{tag}_init.pkl")
        os.makedirs(os.path.dirname(restore), exist_ok=True)
        cmd = [PY, os.path.join(ROOT, "scripts/adapt_ckpt.py"),
               "--in", src, "--out", restore]
        if need_hist:
            cmd += ["--obs-history", args.obs_history]
        if need_scale:
            cmd += ["--action-scale-from", src_scale, "--action-scale-to", st["scale"]]
        if run(cmd, f"段{i} 重みの手術") != 0:
            sys.exit(f"[ERR] adapt_ckpt が失敗した (段 {i})")

    # --- 学習 ---
    # --fresh-critic は付けない。報酬の形は段を通して同じで、critic の重みも
    # 手術でそのまま引き継げているので、捨てると価値推定を毎段ゼロから
    # やり直すことになる (立ち上がりに数十 M step かかる)。
    cmd = [PY, os.path.join(ROOT, "scripts/train.py"),
           "--env", "terrain", "--tag", tag,
           "--difficulty", 1.0,
           "--step-height", st["step"],
           # 足上げ目標は **段差に連動させず据え置く**。terrain_config の既定は
           # SWING_HEIGHT + step_height なので 60mm 地形では 115mm に跳ね上がり、
           # 方策は足上げに過剰投資して胴体の扱いを失う (run d40 の失敗)。
           # 律速が足上げから重心移送に移った後は上げてはいけない (e40 の教訓)。
           "--max-foot-height", 0.085,
           "--gait-freq", 0.7, 1.2,
           "--spawn-radius", 4.0,
           "--action-scale", st["scale"],
           "--obs-history", args.obs_history, "--obs-stride", args.obs_stride,
           "--lr", args.lr,
           "--restore", restore,
           "--num-envs", args.num_envs,
           "--num-timesteps", st["steps"],
           "--num-evals", 40, "--video-every", 5, "--video-secs", 8]
    t0 = time.time()
    if run(cmd, f"段{i} 学習") != 0:
        sys.exit(f"[ERR] train.py が失敗した (段 {i})")
    mins = (time.time() - t0) / 60

    # --- 成績を測る。判断はここの数字で行う ---
    if not args.no_eval:
        run([PY, os.path.join(ROOT, "scripts/step_test.py"),
             "--env", "terrain", "--ckpt", ckpt_of(st["tag"]),
             "--heights", "0,20,30,40,50,60,70", "--trials", 24],
            f"段{i} 越えられる段差")

    manifest.append(dict(stage=i, tag=tag, step_mm=st["step"] * 1000,
                         scale=st["scale"], steps=st["steps"],
                         restore=restore, minutes=round(mins, 1)))

if manifest and not args.dry_run:
    path = os.path.join(ROOT, "out", "train", f"{args.prefix}_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"\n[done] {path}")
    print("  越えられる段差は上の step_test の表で比較すること。"
          "基準は e40 の 40mm (30mm 83.3% / 40mm 70.8% / 50mm 4.2%)")
    print("  実機用の両立版にするには、最終段の重みと u6 を "
          "scripts/distill.py で合流させること (OUTPUTS.md の u_stand の作り方)")
