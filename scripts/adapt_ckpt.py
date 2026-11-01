"""学習済み .pkl を「観測履歴スタック付き」の観測次元に手術して移植する。

env/joystick.py の obs_history を 0 から L に上げると観測次元が変わるので、
38 次元で学習した既存の重み (out/train/u_stand/ckpt/final.pkl など) はそのまま
では読めない。捨てて学習し直すのは論外なので、ここで次元だけ広げて fine-tune を
続けられるようにする。

大前提は **初期状態で元の方策と数値的に完全に等価になること**。そのために:

  - policy の hidden_0.kernel に足した行は全部ゼロ。履歴入力は初期状態で出力に
    一切寄与しないので、行動は元の方策とビット単位に近いレベルで同じになる。
  - critic (value) の観測は privileged_state = hstack([state, 特権情報 70]) なので、
    履歴部は **末尾ではなく index 38 に挿入**される。ゼロ行も同じ位置に挿入する。
    末尾に足すと特権情報 70 次元が 21*L だけずれて critic が壊れる。
  - normalizer (mean/std/summed_variance) の新しい次元は、ゼロや 1 ではなく
    **対応する現在フレームの統計をコピー**する。履歴フレームの中身は現在フレームの
    [gyro, accel*scale, gravity, action] と全く同じ物理量なので、それが正しい初期値。
    ゼロ埋めすると count が既に巨大なため、Welford 更新では何十 M step 経っても
    正しい値まで動かない。

--action-scale-from / --action-scale-to (任意):
  env 側の残差スケール action_scale を A から B に変えると、同じ方策出力でも関節の
  振れ幅が B/A 倍になる。初期の挙動をおおよそ保つため policy 最終層の
  **先頭 12 出力 (行動の平均) だけ** を A/B 倍する (後半 12 は log std なので触らない)。
  ただしこれは tanh の前の値を線形に縮めるだけなので、**|tanh 前の値| が小さい範囲で
  しか一致しない近似的な暖機であって、等価変換ではない**。飽和気味 (|x| が 1 を超える)
  の関節ほどずれる。あくまで fine-tune の初期値を作るためのもの。

使い方 (--obs-history と --action-scale-* はどちらも任意。ただし両方省略は不可):
  python scripts/adapt_ckpt.py --in SRC.pkl --out DST.pkl [--obs-history L]
                               [--action-scale-from A --action-scale-to B]

  --obs-history を省略すると観測次元には一切触らず、残差スケールの付け替えだけを行う
  (入力が 38 次元でも既に拡張済みの次元でも受け付ける)。--obs-history L を指定しても
  入力が既に 38 + L*21 次元なら素通しするので、何度流しても結果は変わらない (冪等)。
"""
import argparse
import dataclasses
import os
import pickle
import sys

import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.joystick import HIST_FRAME_DIM  # noqa: E402  (1 フレーム 21 次元)

# 履歴を持たない素の actor 観測の次元。env/joystick.py の _get_obs と一致させること。
BASE_STATE_DIM = 38
# 履歴 1 フレーム [gyro 3, accel*scale 3, gravity 3, action 12] が、現在フレームの
# 38 次元の中のどこに対応するか。command 3 と phase 2 は履歴に入らないので飛ばす。
#   0-2   gyro
#   3-5   accel*scale
#   6-8   gravity
#   9-11  command      <- 履歴に入らない
#   12-23 last_act     <- 履歴の action はこれと同じ物理量
#   24-35 last_last_act
#   36-37 phase        <- 履歴に入らない
HIST_SRC_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] + list(range(12, 24)))
ACT_DIM = 12


def _silu(x):
    """deploy/numpy_policy.py の _silu と同じ (brax の既定活性化は tanh ではない)。"""
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _forward(params, mean, std, obs):
    """numpy だけで MLP を順伝播する。obs は (N, obs_dim)。"""
    x = (np.asarray(obs, np.float64) - np.asarray(mean, np.float64)) \
        / np.asarray(std, np.float64)
    p = params["params"]
    n = len([k for k in p if k.startswith("hidden_")])
    for i in range(n):
        w = np.asarray(p[f"hidden_{i}"]["kernel"], np.float64)
        b = np.asarray(p[f"hidden_{i}"]["bias"], np.float64)
        x = x @ w + b
        if i < n - 1:
            x = _silu(x)
    return x


def _n_hidden(params):
    return len([k for k in params["params"] if k.startswith("hidden_")])


def _insert_zeros_rows(kernel, at, n_rows):
    """kernel の行方向 index `at` にゼロ行を n_rows 本挿入する。"""
    k = np.asarray(kernel)
    z = np.zeros((n_rows, k.shape[1]), dtype=k.dtype)
    return jnp.asarray(np.concatenate([k[:at], z, k[at:]], axis=0))


def _insert_hist_stats(vec, at, length):
    """統計ベクトルの index `at` に、現在フレームの対応値のコピーを length 回挿入する。"""
    v = np.asarray(vec)
    hist = np.concatenate([v[HIST_SRC_IDX] for _ in range(length)], axis=0)
    return jnp.asarray(np.concatenate([v[:at], hist, v[at:]], axis=0))


def _copy_params(params):
    """{'params': {'hidden_i': {'kernel','bias'}}} を浅くない形で複製する。"""
    return {"params": {k: dict(v) for k, v in params["params"].items()}}


def expand_obs_history(stats, actor, critic, length):
    """観測履歴 L フレーム分だけ観測次元を広げる。追加分の重みはすべてゼロ。"""
    n_new = length * HIST_FRAME_DIM

    state_dim = int(np.asarray(stats.mean["state"]).shape[0])
    priv_dim = int(np.asarray(stats.mean["privileged_state"]).shape[0])
    if state_dim != BASE_STATE_DIM:
        raise SystemExit(
            f"[NG] state が {state_dim} 次元。{BASE_STATE_DIM} 次元 (履歴なし) の "
            f".pkl 以外は受け付けない (既に拡張済みの可能性)")
    a_in = int(np.asarray(actor["params"]["hidden_0"]["kernel"]).shape[0])
    c_in = int(np.asarray(critic["params"]["hidden_0"]["kernel"]).shape[0])
    if a_in != state_dim or c_in != priv_dim:
        raise SystemExit(
            f"[NG] 重みの入力次元 (policy {a_in} / value {c_in}) が normalizer "
            f"(state {state_dim} / privileged {priv_dim}) と合わない")

    # --- normalizer ---
    new_stat_fields = {}
    for field in ("mean", "std", "summed_variance"):
        old = getattr(stats, field)
        new_stat_fields[field] = {
            # state: 38 次元の **後ろ** に連結 (= index 38 に挿入と同じ)
            "state": _insert_hist_stats(old["state"], BASE_STATE_DIM, length),
            # privileged: hstack([state, 特権 70]) なので index 38 に **挿入**
            "privileged_state": _insert_hist_stats(
                old["privileged_state"], BASE_STATE_DIM, length),
        }
    # count / std_eps / mode はそのまま引き継ぐ
    new_stats = dataclasses.replace(stats, **new_stat_fields)

    # --- 重み (hidden_0 の kernel だけ。bias と hidden_1.. は一切触らない) ---
    new_actor = _copy_params(actor)
    new_actor["params"]["hidden_0"]["kernel"] = _insert_zeros_rows(
        actor["params"]["hidden_0"]["kernel"], BASE_STATE_DIM, n_new)
    new_critic = _copy_params(critic)
    new_critic["params"]["hidden_0"]["kernel"] = _insert_zeros_rows(
        critic["params"]["hidden_0"]["kernel"], BASE_STATE_DIM, n_new)
    return new_stats, new_actor, new_critic


def rescale_action(actor, ratio):
    """policy 最終層の先頭 12 出力 (行動の平均) だけを ratio 倍する。"""
    new_actor = _copy_params(actor)
    last = f"hidden_{_n_hidden(actor) - 1}"
    k = np.array(np.asarray(new_actor["params"][last]["kernel"]))
    b = np.array(np.asarray(new_actor["params"][last]["bias"]))
    if b.shape[0] != 2 * ACT_DIM:
        raise SystemExit(
            f"[NG] policy 出力が {b.shape[0]} 次元。平均 {ACT_DIM} + log std "
            f"{ACT_DIM} = {2 * ACT_DIM} を想定している")
    k[:, :ACT_DIM] *= ratio          # 後半 12 (log std) は触らない
    b[:ACT_DIM] *= ratio
    new_actor["params"][last]["kernel"] = jnp.asarray(k)
    new_actor["params"][last]["bias"] = jnp.asarray(b)
    return new_actor


def _pad_hist(obs, at, n_new):
    """観測の index `at` に履歴部 (全ゼロ) を挿入する。"""
    z = np.zeros((obs.shape[0], n_new), dtype=obs.dtype)
    return np.concatenate([obs[:, :at], z, obs[:, at:]], axis=1)


def verify(stats, actor, critic, new_stats, new_actor, new_critic, length,
           n_samples=256, tol=1e-5):
    """履歴部を全ゼロで埋めた観測に対し、拡張前後で出力が一致することを確かめる。"""
    n_new = length * HIST_FRAME_DIM
    rng = np.random.default_rng(0)
    errs = {}
    for name, key, old_p, new_p in (
        ("policy", "state", actor, new_actor),
        ("value", "privileged_state", critic, new_critic),
    ):
        dim = int(np.asarray(stats.mean[key]).shape[0])
        obs = rng.normal(size=(n_samples, dim)).astype(np.float32)
        y_old = _forward(old_p, stats.mean[key], stats.std[key], obs)
        y_new = _forward(new_p, new_stats.mean[key], new_stats.std[key],
                         _pad_hist(obs, BASE_STATE_DIM, n_new))
        err = float(np.abs(y_new - y_old).max())
        errs[name] = err
        tag = "OK" if err < tol else "NG"
        print(f"[{tag}] {name} 出力一致 (最大差 {err:.2e}, 許容 {tol:.0e}, "
              f"{n_samples} サンプル)")
        if err >= tol:
            raise SystemExit(
                f"[NG] {name} の等価性検証に失敗した。書き出しを中止する")
    return errs


def report_rescale(stats_e, actor_e, actor_s, ratio, n_samples=256):
    """スケール付け替え後、行動の平均が概ね ratio 倍になっているかを見る。"""
    rng = np.random.default_rng(1)
    dim = int(np.asarray(stats_e.mean["state"]).shape[0])
    obs = rng.normal(size=(n_samples, dim)).astype(np.float32)
    mean = stats_e.mean["state"]
    std = stats_e.std["state"]
    a0 = _forward(actor_e, mean, std, obs)[:, :ACT_DIM]
    a1 = _forward(actor_s, mean, std, obs)[:, :ACT_DIM]
    pre = float(np.abs(a1 / np.where(np.abs(a0) < 1e-12, 1e-12, a0)).mean())
    t0, t1 = np.tanh(a0), np.tanh(a1)
    sel = np.abs(t0) > 1e-6
    post = float(np.median(t1[sel] / t0[sel]))
    sat = float((np.abs(a0) > 1.0).mean())
    print(f"[info] tanh 前の行動平均の倍率 = {pre:.4f} (目標 {ratio:.4f}、線形なので厳密)")
    print(f"[info] tanh 後 (実際の行動) の倍率の中央値 = {post:.4f} "
          f"(目標 {ratio:.4f}。|tanh 前| > 1 の飽和気味の成分が {sat * 100:.1f}%)")
    print("[warn] スケール付け替えは近似的な暖機であって等価変換ではない。"
          "fine-tune 前提で使うこと")



def main():
    ap = argparse.ArgumentParser(
        description="brax PPO の .pkl を観測履歴スタック付きの次元に移植する")
    ap.add_argument("--in", dest="src", required=True, help="入力 .pkl")
    ap.add_argument("--out", dest="dst", required=True, help="出力 .pkl")
    ap.add_argument("--obs-history", type=int, default=None,
                    help="履歴フレーム数 L (env の obs_history.length と揃える)。"
                         "省略すると観測次元には一切触らない "
                         "(残差スケールの付け替えだけしたいとき用)")
    ap.add_argument("--action-scale-from", type=float, default=None,
                    help="元の env の action_scale A")
    ap.add_argument("--action-scale-to", type=float, default=None,
                    help="新しい env の action_scale B")
    args = ap.parse_args()

    L = args.obs_history
    if L is not None and L < 0:
        raise SystemExit("[NG] --obs-history は 0 以上")
    if (args.action_scale_from is None) != (args.action_scale_to is None):
        raise SystemExit("[NG] --action-scale-from と --action-scale-to は両方指定する")
    if args.action_scale_from is not None and (
            args.action_scale_from <= 0 or args.action_scale_to <= 0):
        raise SystemExit("[NG] action_scale は正の数")
    if L is None and args.action_scale_from is None:
        raise SystemExit("[NG] --obs-history も --action-scale-from/--action-scale-to も"
                         "指定されていない。やることが無い")

    with open(args.src, "rb") as f:
        stats, actor, critic = pickle.load(f)

    state_dim = int(np.asarray(stats.mean["state"]).shape[0])
    priv_dim = int(np.asarray(stats.mean["privileged_state"]).shape[0])
    print(f"[in   ] {args.src}")
    print(f"         state {state_dim} / privileged {priv_dim}")

    # --- 観測次元を広げるかどうかを決める ---
    # --obs-history を省略したときは入力の次元を問わず素通し。指定されていて
    # 入力が既に 38 + L*21 次元なら冪等に素通しする (curriculum の --from-stage で
    # 同じ段をやり直しても二重に広がらないように)。
    do_expand = False
    if L is None:
        print("[expand] --obs-history 省略のため観測次元は変更しない "
              "(入力が何次元でも素通し)")
    elif L == 0:
        print("[expand] --obs-history 0 のため観測次元は変更しない")
    elif state_dim == BASE_STATE_DIM + L * HIST_FRAME_DIM:
        print(f"[expand] state が既に {state_dim} = {BASE_STATE_DIM} + {L}*"
              f"{HIST_FRAME_DIM} 次元。既に拡張済みなので観測次元は素通し")
    elif state_dim == BASE_STATE_DIM:
        do_expand = True
    else:
        raise SystemExit(
            f"[NG] state が {state_dim} 次元。--obs-history {L} が受け付けるのは "
            f"{BASE_STATE_DIM} (未拡張) か "
            f"{BASE_STATE_DIM + L * HIST_FRAME_DIM} (拡張済み) だけ")

    if do_expand:
        new_stats, new_actor, new_critic = expand_obs_history(
            stats, actor, critic, L)
        n_new = L * HIST_FRAME_DIM
        print(f"[expand] 履歴 {L} フレーム x {HIST_FRAME_DIM} 次元 = {n_new} 次元を追加")
        print(f"         state       {state_dim} -> {state_dim + n_new} (末尾に連結)")
        print(f"         privileged  {priv_dim} -> {priv_dim + n_new} "
              f"(index {BASE_STATE_DIM} に挿入。末尾ではない)")
        print(f"         policy hidden_0.kernel {state_dim}x256 -> "
              f"{state_dim + n_new}x256 (ゼロ行 {n_new} 本を末尾に)")
        print(f"         value  hidden_0.kernel {priv_dim}x256 -> "
              f"{priv_dim + n_new}x256 (ゼロ行 {n_new} 本を index "
              f"{BASE_STATE_DIM} に)")
        print(f"         bias と hidden_1.. は変更なし")
        print(f"         normalizer (mean/std/summed_variance) のコピー元 index = "
              f"{HIST_SRC_IDX.tolist()} を {L} 回")
        print(f"         count / std_eps / mode は据え置き")
        # 等価性の検証は「履歴拡張のみ」の中間状態に対して行う
        verify(stats, actor, critic, new_stats, new_actor, new_critic, L)
    else:
        new_stats, new_actor, new_critic = stats, actor, critic
        print("[skip ] 観測次元を変えていないので等価性検証は行わない")

    if args.action_scale_from is not None:
        ratio = args.action_scale_from / args.action_scale_to
        scaled_actor = rescale_action(new_actor, ratio)
        print(f"[scale ] action_scale {args.action_scale_from} -> "
              f"{args.action_scale_to} のため policy 最終層の先頭 {ACT_DIM} 出力 "
              f"(行動の平均) を {ratio:.6f} 倍 (後半 {ACT_DIM} = log std は据え置き)")
        report_rescale(new_stats, new_actor, scaled_actor, ratio)
        new_actor = scaled_actor

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "wb") as f:
        pickle.dump((new_stats, new_actor, new_critic), f)

    out_state = int(np.asarray(new_stats.mean["state"]).shape[0])
    out_priv = int(np.asarray(new_stats.mean["privileged_state"]).shape[0])
    print(f"[out  ] {args.dst}  "
          f"({os.path.getsize(args.dst) / 1024:.0f} KB)")
    print(f"         state {out_state} / privileged {out_priv}")
    if out_state != BASE_STATE_DIM:
        n_hist = (out_state - BASE_STATE_DIM) // HIST_FRAME_DIM
        print(f"         学習再開時は env の obs_history.length={n_hist} と揃えること")


if __name__ == "__main__":
    main()
