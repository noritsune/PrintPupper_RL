"""方策の推論。numpy だけで動く。ラズパイ側はこれを使う。

scripts/policy.py の NumpyPolicy と同じ計算を、.npz (jax/brax 非依存) から
読むようにしただけのもの。踏み外すとエラーにならず「それらしいが全く別の
行動」が静かに出てくる点が2つあるので、向こうのコメントをそのまま引き継ぐ:

  1. 隠れ層の活性化は **silu (swish)**。brax の make_ppo_networks の既定値で
     あって tanh ではない。ここを tanh にすると方策がまるごと別物になる
     (実測: 同じ観測に対し行動が最大 1.4 ずれる)。リポジトリ直下の
     pupper_policy.onnx はこのバグを踏んだままなので使わないこと。
  2. 出力層は 2*12 = 24 次元 (平均 12 + log std 12)。決定的に動かすときは
     NormalTanhDistribution.mode() = tanh(平均) なので、先頭 12 に tanh を
     掛ける。ここは tanh で正しい。

scripts/export_npz.py が .pkl 版と数値を突き合わせてから書き出す。
"""
import numpy as np


def _silu(x):
    # exp のオーバーフロー警告を避ける。x が十分小さいところでは silu -> 0。
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class NpzPolicy:
    """export_npz.py が書いた .npz を読んで推論する。"""

    def __init__(self, path, obs_dim=None):
        z = np.load(path)
        self.mean = z["mean"].astype(np.float32)
        self.std = z["std"].astype(np.float32)
        nl = len([k for k in z.files if k.startswith("w")])
        self.layers = [(z[f"w{i}"].astype(np.float32),
                        z[f"b{i}"].astype(np.float32)) for i in range(nl)]
        self.obs_dim = int(self.mean.shape[0])
        self.act_dim = int(self.layers[-1][1].shape[0]) // 2
        if obs_dim is not None:
            assert self.obs_dim == obs_dim, \
                f"npz の観測次元 {self.obs_dim} が {obs_dim} と違う"

    def __call__(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        for w, b in self.layers[:-1]:
            x = _silu(x @ w + b)
        w, b = self.layers[-1]
        x = x @ w + b                       # (24,) = 平均 12 + log std 12
        return np.tanh(x[:self.act_dim])    # = NormalTanhDistribution.mode()
