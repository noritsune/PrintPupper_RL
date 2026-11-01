"""歩行方策 (e40) と復帰方策 (rec9) を 1 つのネットワークに蒸留する。

なぜ必要か: 両立版 (env/unified.py) の観測・可動域は、正立側が e40 と、転倒側が
rec9 と数値的に一致するように作ってある。それでも **どちらか一方の重みから
PPO を始めると、もう一方の技能はゼロから学び直しになる**。実測 (両立環境で):

    e40  の重み: 段差 40mm を 29.2% で越える / 復帰 0.0%
    rec9 の重み: 段差 40mm を  0.0% で越える (そもそも前に進まない) / 復帰 46%

歩行 (v2 -> v3 -> d30b -> e40) も復帰 (rec1 -> rec8) も、それぞれ数百 M step を
かけて到達したものなので、どちらを捨てても高くつく。そこで PPO の前に、
**2 つの教師の行動を 1 つの方策に回帰させる** (行動クローニング)。

  - 正立している状態 (ゲート a > 0.9) では e40 の出力を教師にする。
  - 転倒から立ち上がるまでの軌道は rec9 の出力を教師にする。
    こちらは速度コマンドを 0 に固定して集める (rec9 の学習時と同じ入力分布)。

回帰の対象は方策ネットワークの出力 24 次元そのもの (平均 12 + log std 12)。
tanh の後ではなく前を合わせるので、飽和域でも勾配が消えない。

正規化器 (running_statistics) は **e40 のものをそのまま使う**。学生の初期値も
e40 なので、正立側は初期時点で e40 と 1 ビットも違わない状態から始まる。
転倒側の観測は e40 の正規化の想定外に出るが、PPO 側で正規化器は更新され続ける。

出力は scripts/train.py --restore がそのまま食える 3 つ組
(normalizer, actor, critic)。critic は新しい報酬に対して無意味なので初期化して
入れてある (--fresh-critic を付けるのと同じ扱い)。

使い方:
    python scripts/distill.py --out out/train/u_distill/ckpt/final.pkl
    python scripts/train.py --env unified --tag u2 \\
        --restore out/train/u_distill/ckpt/final.pkl --fresh-critic ...
"""
import argparse, os, pickle, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--walk", default=os.path.join(ROOT, "out/train/e40/ckpt/final.pkl"),
                help="歩行の教師 (段差歩行の方策)")
ap.add_argument("--recover", default=os.path.join(ROOT, "out/train/rec9/ckpt/final.pkl"),
                help="復帰の教師 (転倒復帰の方策)")
ap.add_argument("--out", default=os.path.join(ROOT, "out/train/u_distill/ckpt/final.pkl"))
ap.add_argument("--envs", type=int, default=256, help="同時に回す env 数")
ap.add_argument("--walk-steps", type=int, default=600, help="歩行データの収集ステップ")
ap.add_argument("--rec-steps", type=int, default=400, help="復帰データの収集ステップ")
ap.add_argument("--rounds", type=int, default=3,
                help="収集のラウンド数 (seed を変えて繰り返す)")
ap.add_argument("--iters", type=int, default=4000, help="回帰の反復回数")
ap.add_argument("--dagger", type=int, default=0, metavar="N",
                help="行動クローニングの後、生徒自身の軌道に教師がラベルを"
                     "付けたデータを足して学習し直す回数 (DAgger)。"
                     "教師の軌道だけで学ぶと誤差が積み上がり、生徒は教師より"
                     "安全側の別物になる。2-3 回で効く")
ap.add_argument("--still-steps", type=int, default=0, metavar="N",
                help="静止指令 (コマンド 0) のときの教師データを N ステップ集める。"
                     "教師は **出力 0 = 基準姿勢**。基準姿勢を保持するだけで "
                     "4脚接地・161.7mm・速度 0.0000 m/s で静止することを実測して"
                     "あるので、これが「起きたらその場で待機」の正解になる。"
                     "0 で無効 (歩行教師が静止指令の状態も受け持つ)")
ap.add_argument("--dagger-iters", type=int, default=6000,
                help="DAgger 1 周あたりの回帰の反復回数")
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--difficulty", type=float, default=1.0)
ap.add_argument("--contacts-per-env", type=int, default=32)
ap.add_argument("--handoff-gate", type=float, default=0.9,
                help="正立ゲートがこれを超えた状態は歩行教師に任せる "
                     "(復帰教師から採らない)。0.9 は可動域が 0.25rad に"
                     "戻りきる点で、env/unified.py の gate_hi と同じ")
args = ap.parse_args()

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".6")

from env import _compat; _compat.install()
import jax
import jax.numpy as jp
import optax

from env.unified import PupperUnified, unified_config
from scripts.policy import make_network


def teacher_fn(net, params):
    """教師の出力 (24 次元: 平均 12 + log std 12) をそのまま返す関数。

    tanh を掛けた行動ではなく **分布のパラメータ** を教師にする。飽和域
    (|action| -> 1) では tanh の勾配が消えるので、そこを合わせにいくと学習が
    進まない。教師の正規化器は教師自身のものを使う (params[0])。
    """
    stats, actor = params[0], params[1]
    return jax.jit(lambda obs: net.policy_network.apply(stats, actor, obs))


def collect(env, net, params, steps, zero_cmd, seed, gate_min=None, gate_max=None,
            actor_params=None, cmd_min=None, zero_target=False):
    """ロールアウトして (state 観測, privileged 観測, 教師出力) を集める。

    actor_params を渡すと **そちらで動かし、教師には正解を付けさせるだけ**に
    なる (DAgger)。行動クローニングは教師の軌道しか見ないので、生徒がわずかに
    ずれた先での正しい行動を知らない。その誤差が積み上がると、生徒は教師より
    「安全側」の別の方策に収束する。実測 (段差 40mm 地形を歩かせたときの
    水平度の 5%点): e40 が +0.779 まで胴体を傾けるのに対し、蒸留した生徒は
    +0.951 までしか傾けない = **段を越えるのに要る姿勢変化を避けている**。
    生徒自身が通る状態を教師にラベルさせれば、そこが埋まる。
    """
    from env import constants as consts
    mm = env.mj_model
    UP = mm.sensor_adr[mm.sensor(consts.UPVECTOR_SENSOR).id] + 2   # upvector z
    fwd = teacher_fn(net, params)
    act_fn = fwd if actor_params is None else teacher_fn(net, actor_params)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    B = args.envs
    st = reset(jax.random.split(jax.random.PRNGKey(seed), B))
    S, P, Y, G, C = [], [], [], [], []
    for k in range(steps):
        if zero_cmd:
            st.info["command"] = jp.zeros((B, 3))
        out = fwd(st.obs)                      # 教師の正解 (学習の目標)
        drive = out if actor_params is None else act_fn(st.obs)   # 実際に動かす方
        S.append(np.asarray(st.obs["state"]))
        P.append(np.asarray(st.obs["privileged_state"]))
        Y.append(np.asarray(out))
        # 正立ゲート a。どの状態をどちらの教師に任せるかの仕分けに使う。
        G.append(np.asarray(env._gate(st.data.sensordata[:, UP])))
        C.append(np.linalg.norm(np.asarray(st.info["command"]), axis=-1))
        act = jp.tanh(drive[..., :env.action_size])     # 決定的に回す
        st = step(st, act)
    S = np.concatenate(S); P = np.concatenate(P)
    Y = np.concatenate(Y); G = np.concatenate(G); C = np.concatenate(C)
    m = np.ones(len(G), bool)
    if gate_min is not None:
        m &= G >= gate_min
    if gate_max is not None:
        m &= G <= gate_max
    if cmd_min is not None:
        m &= C >= cmd_min
    S, P, Y = S[m], P[m], Y[m]
    if zero_target:
        # **目標は「基準姿勢のまま動くな」**。行動の平均 (前半 nu) を 0 にする。
        # log std (後半) は教師のものをそのまま残す。
        Y = Y.copy()
        Y[:, :env.action_size] = 0.0
    return S, P, Y


def main():
    # --- 環境 (収集用。歩行は正立スポーン、復帰は転倒スポーン) ---
    cw = unified_config(); cw.recover.p_walk = 1.0
    cw.terrain.difficulty = args.difficulty
    cw.naconmax = args.contacts_per_env * args.envs
    env_w = PupperUnified(config=cw)
    env_w._mjx_model = env_w._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})

    cr = unified_config(); cr.recover.p_walk = 0.0
    cr.terrain.difficulty = args.difficulty
    cr.naconmax = args.contacts_per_env * args.envs
    cr.episode_length = args.rec_steps
    env_r = PupperUnified(config=cr)
    env_r._mjx_model = env_r._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})

    net = make_network(env_w)
    p_walk = pickle.load(open(args.walk, "rb"))
    p_rec = pickle.load(open(args.recover, "rb"))

    Sw = Pw = Yw = Sr = Pr = Yr = None
    for r in range(args.rounds):
        # 歩行: 明確に正立している状態だけを採る。転んだ後の状態まで e40 に
        # 教えさせると、転倒側に「歩こうとする」教師が混ざってしまう。
        # 歩行: **動けと指示されている状態だけ**。静止指令の状態まで歩行教師に
        # 任せると、静止教師と担当が重なって「どちらでもない行動」になる
        # (教師を分けるときの原則。OUTPUTS.md の「効いた一手」)。
        a = collect(env_w, net, p_walk, args.walk_steps, False, 100 + r,
                    gate_min=args.handoff_gate,
                    cmd_min=0.02 if args.still_steps else None)
        # 復帰: **起き上がるまで** (ゲートが開いている間) だけを採る。
        # 立った後まで復帰教師に任せると、その教師の「立ち方」がそのまま
        # 入ってくる。run u6 は胴体高 180mm・接地脚 2/4 で立つので、歩行
        # 方策が前提とする基準姿勢 (163mm・4脚接地) とずれ、引き渡した
        # 瞬間に歩けない。正立側は歩行教師 (e40) に一本化する。
        b = collect(env_r, net, p_rec, args.rec_steps, True, 200 + r,
                    gate_max=args.handoff_gate)
        if args.still_steps:
            # 静止: 正立していてコマンドが 0 の状態。教師は出力 0。
            c = collect(env_w, net, p_walk, args.still_steps, True, 500 + r,
                        gate_min=args.handoff_gate, zero_target=True)
            a = [np.concatenate([x, y]) for x, y in zip(a, c)]
        Sw, Pw, Yw = [x if y is None else np.concatenate([y, x])
                      for x, y in zip(a, (Sw, Pw, Yw))]
        Sr, Pr, Yr = [x if y is None else np.concatenate([y, x])
                      for x, y in zip(b, (Sr, Pr, Yr))]
        print(f"[collect] round {r}: 歩行 {len(Sw):,} 本 / 復帰 {len(Sr):,} 本", flush=True)

    S = np.concatenate([Sw, Sr]); Y = np.concatenate([Yw, Yr])
    print(f"[data] 合計 {len(S):,} サンプル (歩行 {len(Sw):,} / 復帰 {len(Sr):,})")

    # --- 学生: e40 の正規化器と重みから始める ---
    stats = p_walk[0]
    actor = p_walk[1]
    P = np.concatenate([Pw, Pr])
    S_j, P_j, Y_j = jp.asarray(S), jp.asarray(P), jp.asarray(Y)

    # brax は preprocess_observations_fn を **観測 dict 全体** に掛けてから
    # policy_obs_key を取り出す。state だけ渡すと正規化器の木構造と合わずに
    # 落ちるので、critic 用の観測も一緒に渡す (方策自体は state しか見ない)。
    def loss_fn(a, s, pv, y):
        pred = net.policy_network.apply(
            stats, a, {"state": s, "privileged_state": pv})
        return jp.mean(jp.square(pred - y))

    opt = optax.adam(args.lr)
    opt_state = opt.init(actor)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update(a, o, s, pv, y):
        l, g = grad_fn(a, s, pv, y)
        upd, o = opt.update(g, o)
        return optax.apply_updates(a, upd), o, l

    key = jax.random.PRNGKey(0)

    def fit(iters, tag):
        """いまのデータ全部で回帰する。"""
        nonlocal actor, opt_state, key
        n = len(S_j)
        for it in range(iters):
            key, k = jax.random.split(key)
            idx = jax.random.randint(k, (args.batch,), 0, n)
            actor, opt_state, l = update(actor, opt_state,
                                         S_j[idx], P_j[idx], Y_j[idx])
            if it % 500 == 0 or it == iters - 1:
                lw = float(loss_fn(actor, jp.asarray(Sw[:8192]), jp.asarray(Pw[:8192]),
                                   jp.asarray(Yw[:8192])))
                lr_ = float(loss_fn(actor, jp.asarray(Sr[:8192]), jp.asarray(Pr[:8192]),
                                    jp.asarray(Yr[:8192])))
                print(f"  {tag}[{it:5d}] loss {float(l):.4f}  歩行 {lw:.4f}  "
                      f"復帰 {lr_:.4f}", flush=True)

    fit(args.iters, "BC ")

    # --- DAgger: 生徒自身が通る状態に教師がラベルを付ける ---
    for dr in range(args.dagger):
        student = (stats, actor)
        a = collect(env_w, net, p_walk, args.walk_steps, False, 300 + dr,
                    gate_min=args.handoff_gate, actor_params=student,
                    cmd_min=0.02 if args.still_steps else None)
        if args.still_steps:
            c = collect(env_w, net, p_walk, args.still_steps, True, 700 + dr,
                        gate_min=args.handoff_gate, actor_params=student,
                        zero_target=True)
            a = [np.concatenate([x, y]) for x, y in zip(a, c)]
        b = collect(env_r, net, p_rec, args.rec_steps, True, 400 + dr,
                    gate_max=args.handoff_gate, actor_params=student)
        Sw, Pw, Yw = [np.concatenate([y, x]) for x, y in zip(a, (Sw, Pw, Yw))]
        Sr, Pr, Yr = [np.concatenate([y, x]) for x, y in zip(b, (Sr, Pr, Yr))]
        S_j = jp.asarray(np.concatenate([Sw, Sr]))
        P_j = jp.asarray(np.concatenate([Pw, Pr]))
        Y_j = jp.asarray(np.concatenate([Yw, Yr]))
        print(f"[dagger {dr}] 生徒の軌道を追加 -> 歩行 {len(Sw):,} / "
              f"復帰 {len(Sr):,} 本", flush=True)
        fit(args.dagger_iters, f"DA{dr} ")

    # --- critic は新しい報酬に対して無意味なので初期化して入れる ---
    key, k = jax.random.split(key)
    critic = net.value_network.init(k)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump((stats, actor, critic), f)
    print(f"[done] -> {args.out}")
    print("  次: python scripts/train.py --env unified --tag <tag> "
          f"--restore {args.out} --fresh-critic ...")


if __name__ == "__main__":
    main()
