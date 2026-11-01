"""brax 0.14.2 と jax >= 0.11 の互換シム。

jax 0.11 で jax.device_put_replicated が削除されたが、brax の PPO 実装
(training/pmap.py と agents/ppo/train.py) がまだ呼んでいる。brax 側は既に
「新しい jax.pmap」に対応済みでこの1関数だけが欠けているため、公式の
移行ガイドどおりの置き換えを差し戻す。
"""
import jax
import jax.numpy as jp
import numpy as np


def install():
    if hasattr(jax, "device_put_replicated"):
        return False

    def device_put_replicated(x, devices):
        n = len(devices)
        stacked = jax.tree_util.tree_map(lambda y: jp.stack([jp.asarray(y)] * n), x)
        if n == 1:
            return jax.device_put(stacked, devices[0])
        mesh = jax.sharding.Mesh(np.asarray(devices), ("_dev",))
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("_dev"))
        return jax.device_put(stacked, sharding)

    jax.device_put_replicated = device_put_replicated
    return True
