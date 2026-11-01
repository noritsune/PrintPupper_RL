"""学習済み方策 (.pkl) を ONNX に書き出す。実機 (ROS2 + onnxruntime) 用。

観測の正規化をモデルの中に埋め込むので、実機側は env/deploy.py の ObsBuilder
が作った 38 次元をそのまま食わせるだけでよい。

注意 (ここを外すとエラーは出ないまま方策が別物になる):
  - 隠れ層の活性化は **silu (swish)**。brax の make_ppo_networks の既定値。
    以前このスクリプトは tanh を使っており、書き出した ONNX は学習した方策と
    別物だった (同じ観測に対し行動が最大 1.4 ずれる)。
  - 正規化は brax と同じ (x - mean) / std。std は brax 側で 1e-6 を下限に
    クリップ済みなので、ここで更に 1e-6 を足してはいけない (位相のような
    定数入力の値が半分になる)。
  - 出力は行動平均 12 次元のみ (tanh は掛けない)。実機側で tanh を掛けて
    NormalTanhDistribution.mode() と一致させる。env/deploy.py と
    scripts/view.py の OnnxPolicy はそうしている。

書き出したあと、numpy 実装 (scripts/policy.py の NumpyPolicy) と数値を
突き合わせて一致を確認する。
"""
import argparse, os, pickle, sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from policy import NumpyPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ROOT, "out/train/bal1/ckpt/final.pkl"))
ap.add_argument("--out", default=os.path.join(ROOT, "pupper_policy.onnx"))
args = ap.parse_args()

with open(args.ckpt, "rb") as f:
    stats, actor_params, _ = pickle.load(f)
mean_np = np.array(stats.mean["state"], dtype=np.float32)
std_np = np.array(stats.std["state"], dtype=np.float32)
p = actor_params["params"]
sizes = []
i = 0
while f"hidden_{i}" in p:
    sizes.append(np.array(p[f"hidden_{i}"]["kernel"]).shape)
    i += 1
OBS, ACT = sizes[0][0], sizes[-1][1]


class PupperPolicy(nn.Module):
    def __init__(self, mean, std, sizes):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))
        self.hidden = nn.ModuleList(
            [nn.Linear(i, o) for i, o in sizes])

    def forward(self, x):
        x = (x - self.mean) / self.std          # running_statistics.normalize
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden) - 1:
                x = torch.nn.functional.silu(x)  # brax の既定。tanh ではない
        return x[:, :ACT // 2]                   # 行動平均だけ (tanh は実機側)


policy = PupperPolicy(mean_np, std_np, sizes)
with torch.no_grad():
    for i, layer in enumerate(policy.hidden):
        # Flax の kernel は (In, Out)、PyTorch の weight は (Out, In)
        layer.weight.copy_(torch.from_numpy(np.array(p[f"hidden_{i}"]["kernel"])).T)
        layer.bias.copy_(torch.from_numpy(np.array(p[f"hidden_{i}"]["bias"])))
policy.eval()

torch.onnx.export(
    policy, torch.randn(1, OBS, dtype=torch.float32), args.out,
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
)
print(f"[export] {args.ckpt}\n     -> {args.out}  "
      f"obs={OBS} hidden={[s[1] for s in sizes[:-1]]} out={ACT}")

# --- 書き出したものを numpy 実装と突き合わせる ---
ref = NumpyPolicy(args.ckpt)
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(64):
        o = rng.normal(size=OBS).astype(np.float32)
        y = sess.run(None, {"obs": o[None]})[0][0]
        err = max(err, np.abs(np.tanh(y) - ref(o)).max())
    print(f"[check ] numpy 実装との最大差 (tanh 後の行動) {err:.2e}  "
          f"{'OK' if err < 1e-4 else 'NG: 活性化か正規化がずれている'}")
    sys.exit(0 if err < 1e-4 else 1)
except ImportError:
    print("[check ] onnxruntime が無いので検証をとばした")
