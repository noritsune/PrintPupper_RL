"""学習済みポリシーの読み込みと推論。評価・可視化・書き出しで共用する。"""
import functools, os, pickle, sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from env.joystick import PupperJoystick, default_config, HIST_FRAME_DIM

# scripts/train.py の network_factory と必ず一致させること
HIDDEN = (256, 128, 64)


def ckpt_obs_history(ckpt) -> int:
    """.pkl の観測次元から、観測履歴スタックのフレーム数 L を逆算する。

    観測は 38 + L*21 次元 (env/joystick.py) なので L は一意に決まる。
    評価スクリプトが env を組むとき、**学習時と同じ L を与えないと
    次元不一致で落ちる**。手で渡させると必ず取り違えるので、重みから読む。

    stride はこの次元からは復元できない (L と stride の積が窓幅)。
    既定の 4 以外を使った run では明示的に渡すこと。
    """
    params = pickle.load(open(ckpt, "rb")) if isinstance(ckpt, str) else ckpt
    dim = int(params[0].mean["state"].shape[0])
    extra = dim - 38
    if extra < 0 or extra % HIST_FRAME_DIM:
        raise SystemExit(
            f"観測次元 {dim} が 38 + L*{HIST_FRAME_DIM} の形になっていない")
    return extra // HIST_FRAME_DIM


def make_network(env, normalize_observations=True):
    # brax の ppo.train は normalize_observations=True のとき
    # preprocess_observations_fn=running_statistics.normalize を渡す。
    # ここで渡し忘れると方策は「正規化前の生の観測」を受け取ることになり、
    # 重みと入力のスケールが噛み合わず出力が意味をなさなくなる。
    # (学習は正常なのに評価だけ動かない、という形で表面化する)
    preprocess = (running_statistics.normalize if normalize_observations
                  else lambda x, y: x)
    return ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess,
        policy_hidden_layer_sizes=HIDDEN,
        value_hidden_layer_sizes=HIDDEN,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )


class NumpyPolicy:
    """brax PPO の ckpt (.pkl) を numpy だけで動かす。jax も brax も要らない。

    実機 (ROS2 ノード) と scripts/view.py がこれを使う。brax 側と完全に一致
    させるために踏むべき点が2つある。踏み外すとエラーにはならず、
    「それらしいが全く別の行動」が静かに出てくる。

      1. 隠れ層の活性化は **silu (swish)**。brax の make_ppo_networks の既定値で
         あって tanh ではない。ここを tanh にすると学習済みの方策がまるごと
         別物になる (実測: 同じ観測に対し行動が最大 1.4 ずれ、5分立てる方策が
         0.5秒で転ぶ)。
      2. 出力層は 2*12 = 24 次元 (平均 12 + log std 12)。決定的に動かすときは
         NormalTanhDistribution.mode() = tanh(平均) なので、先頭 12 に tanh を
         掛ける。ここは tanh で正しい。

    scripts/check_deploy.py --ckpt が brax の推論と数値で突き合わせる。
    """

    def __init__(self, path, obs_dim=None):
        import pickle
        stats, actor, _ = pickle.load(open(path, "rb"))
        self.mean = np.asarray(stats.mean["state"], np.float32)
        self.std = np.asarray(stats.std["state"], np.float32)
        p = actor["params"]
        nl = len([k for k in p if k.startswith("hidden_")])
        self.layers = [(np.asarray(p[f"hidden_{i}"]["kernel"], np.float32),
                        np.asarray(p[f"hidden_{i}"]["bias"], np.float32))
                       for i in range(nl)]
        if obs_dim is not None:
            assert self.mean.shape == (obs_dim,), \
                f"ckpt の観測次元 {self.mean.shape} が {obs_dim} と違う"

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    def __call__(self, obs):
        x = (obs - self.mean) / self.std       # running_statistics.normalize
        for w, b in self.layers[:-1]:
            x = self._silu(x @ w + b)
        w, b = self.layers[-1]
        x = x @ w + b                          # (24,) = 平均 12 + log std 12
        return np.tanh(x[:12])                 # = NormalTanhDistribution.mode()


def load(ckpt, env=None, config_overrides=None, deterministic=True):
    """(env, inference_fn, params) を返す。"""
    cfg = default_config()
    if config_overrides:
        for k, v in config_overrides.items():
            path = k.split(".")
            node = cfg
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] = v
    env = env or PupperJoystick(config=cfg)
    net = make_network(env)
    params = pickle.load(open(ckpt, "rb"))
    make_policy = ppo_networks.make_inference_fn(net)
    return env, jax.jit(make_policy(params, deterministic=deterministic)), params


def rollout(env, inference, n_steps, command, seed=0, model_patch=None,
            push=None, collect=None):
    """1 エピソード回して記録を返す。

    command: (3,) 固定、または step 番号 -> (3,) の関数
    push:    step 番号 -> (3,) 速度キック [m/s]、None で無外乱
    """
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    rec = {k: [] for k in ("qpos", "qvel", "cmd", "act", "tau", "contact",
                           "foot_z", "local_v", "gyro", "grav", "joint_q",
                           "joint_v", "done")}
    from env import constants as consts
    for k in range(n_steps):
        c = command(k) if callable(command) else command
        state.info["command"] = jp.array(c, jp.float32)
        if push is not None:
            dv = push(k)
            if dv is not None:
                state = state.replace(
                    data=state.data.replace(qvel=state.data.qvel.at[:3].add(jp.array(dv))))
        act, _ = inference(state.obs, jax.random.PRNGKey(k))
        state = step(state, act)
        d = state.data
        rec["qpos"].append(np.array(d.qpos)); rec["qvel"].append(np.array(d.qvel))
        rec["cmd"].append(np.array(c)); rec["act"].append(np.array(act))
        rec["tau"].append(np.array(d.actuator_force))
        rec["contact"].append(np.array(d.sensordata[env._feet_contact_adr] > 0))
        rec["foot_z"].append(np.array(d.site_xpos[env._feet_site_id][:, 2]))
        rec["local_v"].append(np.array(env._sensor(d, consts.LOCAL_LINVEL_SENSOR)))
        rec["gyro"].append(np.array(env._sensor(d, consts.GYRO_SENSOR)))
        rec["grav"].append(np.array(env._gravity(d)))
        rec["joint_q"].append(np.array(d.qpos[7:]))
        rec["joint_v"].append(np.array(d.qvel[6:]))
        rec["done"].append(float(state.done))
        if collect is not None:
            collect(state)
    return {k: np.array(v) for k, v in rec.items()}
