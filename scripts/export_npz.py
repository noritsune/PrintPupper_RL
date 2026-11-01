"""学習済み方策 (.pkl) を jax/brax 非依存の .npz に書き出す。実機 (ラズパイ) 用。

ckpt の pickle は brax の running_statistics 型を含むので、brax の入っていない
ラズパイでは unpickle できない。ここで numpy 配列だけに剥がす。

出力の中身 (deploy/numpy_policy.py が読む):
  mean, std        観測 38 次元の正規化統計 (running_statistics.normalize)
  w0..wN, b0..bN   隠れ層の重み。kernel は Flax のまま (In, Out)

書き出したあと、scripts/policy.py の NumpyPolicy (= 学習側と 1.5e-07 で一致する
ことが check_deploy.py で確認済みの実装) と数値を突き合わせる。
"""
import argparse, os, pickle, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "out/train/u_stand/ckpt/final.pkl"))
ap.add_argument("--out", default=os.path.join(ROOT, "deploy/policy_u_stand.npz"))
ap.add_argument("--samples", type=int, default=1000,
                help="NumpyPolicy との照合に使うランダム観測の本数")
args = ap.parse_args()

with open(args.ckpt, "rb") as f:
    stats, actor, _ = pickle.load(f)

out = {"mean": np.asarray(stats.mean["state"], np.float32),
       "std": np.asarray(stats.std["state"], np.float32)}
p = actor["params"]
nl = len([k for k in p if k.startswith("hidden_")])
for i in range(nl):
    out[f"w{i}"] = np.asarray(p[f"hidden_{i}"]["kernel"], np.float32)
    out[f"b{i}"] = np.asarray(p[f"hidden_{i}"]["bias"], np.float32)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez(args.out, **out)

obs_dim = out["mean"].shape[0]
sizes = [out[f"w{i}"].shape for i in range(nl)]
print(f"[export] {args.ckpt}\n     -> {args.out}  "
      f"obs={obs_dim} layers={[s[1] for s in sizes]} "
      f"({os.path.getsize(args.out) / 1024:.0f} KB)")

# --- 書き出したものを NumpyPolicy (.pkl 版) と突き合わせる ---
# ここがずれていたら実機に持っていっても意味がないので、書き出しと同じ場所で検査する。
from policy import NumpyPolicy

sys.path.insert(0, os.path.join(ROOT, "deploy"))
from numpy_policy import NpzPolicy

ref = NumpyPolicy(args.ckpt)
new = NpzPolicy(args.out)
rng = np.random.default_rng(0)
err = 0.0
for _ in range(args.samples):
    o = rng.normal(size=obs_dim).astype(np.float32)
    err = max(err, float(np.abs(new(o) - ref(o)).max()))
ok = err < 1e-6
print(f"[check ] NumpyPolicy (.pkl) との最大差 {err:.2e}  "
      f"{'OK' if ok else 'NG: 重みか正規化の取り出しがずれている'}")
sys.exit(0 if ok else 1)
