"""ドメインランダム化。実機転移を狙うなら必須。

PrintPupper 固有の事情:
  - 慣性は実測ではなくメッシュ形状からの推定なので、質量のばらつきは大きめに取る。
  - PWM サーボは個体差・電圧降下でゲインとトルクが大きく変わる。
    DS3235 は 4.8-6V で 30kg-cm (2.94Nm) / 6.8V で 35kg-cm (3.43Nm)。
    電池消耗と個体差を含めて 2.4-3.4Nm を掃く。
  - トルクと無負荷速度は独立ではない。電圧が下がれば両方下がる。damping は
    tau_stall/w_noload なので、トルクを振ったら damping も整合的に振る。
"""
import jax
import jax.numpy as jp


def domain_randomize(model, rng):
    @jax.vmap
    def rand(rng):
        # 床の摩擦 0.4-1.2 (フローリング〜カーペット)
        rng, k = jax.random.split(rng)
        friction = jax.random.uniform(k, (), minval=0.4, maxval=1.2)
        geom_friction = model.geom_friction.at[:, 0].set(friction)

        # 胴体質量 +-20% (バッテリ・配線・追加IMU の不確かさ)
        rng, k = jax.random.split(rng)
        dmass = jax.random.uniform(k, (), minval=-0.2, maxval=0.2)
        body_mass = model.body_mass.at[1].set(model.body_mass[1] * (1 + dmass))

        # 全リンク質量 +-10% (慣性がメッシュ推定である不確かさ)
        rng, k = jax.random.split(rng)
        body_mass = body_mass * jax.random.uniform(
            k, (model.nbody,), minval=0.9, maxval=1.1)

        # 胴体の重心位置 +-8mm (電池搭載位置のずれ)
        rng, k = jax.random.split(rng)
        body_ipos = model.body_ipos.at[1].add(
            jax.random.uniform(k, (3,), minval=-0.008, maxval=0.008))

        # サーボ P ゲイン +-30% (個体差)
        rng, k = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            k, (model.nu,), minval=0.7, maxval=1.3)
        gain = model.actuator_gainprm.at[:, 0].set(kp)
        bias = model.actuator_biasprm.at[:, 1].set(-kp)

        # サーボ最大トルク 2.4-3.4 Nm (電圧 4.8-6.8V + 個体差)
        rng, k = jax.random.split(rng)
        fr = jax.random.uniform(k, (), minval=2.4, maxval=3.4)
        forcerange = model.actuator_forcerange.at[:, 0].set(-fr)
        forcerange = forcerange.at[:, 1].set(fr)

        # 無負荷速度 6.0-8.2 rad/s (DS3235: 0.13-0.17 s/60deg)。
        # damping = tau_stall / w_noload でトルクと整合させる。さらにギヤの
        # 個体差ぶん +-25% を乗せる。
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        w0 = jax.random.uniform(k3, (), minval=6.0, maxval=8.2)
        damping = model.dof_damping.at[6:].set(
            (fr / w0) * jax.random.uniform(
                k1, (model.nu,), minval=0.75, maxval=1.25))
        frictionloss = model.dof_frictionloss.at[6:].set(
            model.dof_frictionloss[6:] * jax.random.uniform(
                k2, (model.nu,), minval=0.5, maxval=2.0))
        return (geom_friction, body_mass, body_ipos, gain, bias,
                forcerange, damping, frictionloss)

    (geom_friction, body_mass, body_ipos, gain, bias,
     forcerange, damping, frictionloss) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "body_mass": 0, "body_ipos": 0,
        "actuator_gainprm": 0, "actuator_biasprm": 0,
        "actuator_forcerange": 0, "dof_damping": 0, "dof_frictionloss": 0,
    })
    model = model.tree_replace({
        "geom_friction": geom_friction, "body_mass": body_mass,
        "body_ipos": body_ipos, "actuator_gainprm": gain,
        "actuator_biasprm": bias, "actuator_forcerange": forcerange,
        "dof_damping": damping, "dof_frictionloss": frictionloss,
    })
    return model, in_axes
