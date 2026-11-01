"""方策の探索ノイズ (log_std) を書き換える。行動クローニングの直後に使う。

なぜ要るか: brax の PPO は tanh をかける前の行動を N(mean, std) から引く。
std が大きいと **1 ステップごとに方策の意図から大きく外れた行動**が出る。
歩行は多少ずれても転ばないが、転倒復帰は数百 ms の蹴り出しの精度で決まる
ので、蒸留した方策をそのまま PPO にかけると **復帰だけが先に壊れる**。

実測 (OUTPUTS.md「段差歩行と転倒復帰を1つの方策に統合する」):
  run u5 は蒸留直後の std 0.49-0.92 のまま PPO を回し、最初の 5M step で
  正立時間が 84% -> 64% に落ちた。std を 0.2 に絞り、--lr 1e-4 /
  --entropy 0.001 と合わせた run u6 は 84% -> 96% に伸びた。

log_std ヘッド (出力層の後半 nu 列) の重みを 0、bias を log(std) にする。
観測に依存しない一定のノイズになる。PPO の entropy 項はここを押し戻すので、
--entropy も下げて使うこと。

使い方:
    python scripts/set_policy_std.py --in out/train/u_distill2/ckpt/final.pkl \
        --out out/train/u_distill3/ckpt/final.pkl --std 0.2
"""
import argparse, os, pickle, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True, help="元の ckpt (.pkl)")
ap.add_argument("--out", dest="dst", required=True)
ap.add_argument("--std", type=float, default=0.2, help="探索ノイズの標準偏差")
ap.add_argument("--nu", type=int, default=12, help="行動次元")
args = ap.parse_args()

import jax.numpy as jp

stats, actor, critic = pickle.load(open(args.src, "rb"))
w = actor["params"]
last = f"hidden_{len([k for k in w if k.startswith('hidden_')]) - 1}"
k = np.asarray(w[last]["kernel"]).copy()
b = np.asarray(w[last]["bias"]).copy()
assert k.shape[1] == 2 * args.nu, f"出力層が {k.shape[1]} 次元。2*nu を想定"
print(f"変更前 std [{np.exp(b[args.nu:]).min():.3f}, {np.exp(b[args.nu:]).max():.3f}]")
k[:, args.nu:] = 0.0
b[args.nu:] = np.log(args.std)
w[last]["kernel"] = jp.asarray(k)
w[last]["bias"] = jp.asarray(b)
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
pickle.dump((stats, actor, critic), open(args.dst, "wb"))
print(f"変更後 std {args.std:.3f} (観測に依存しない定数) -> {args.dst}")
