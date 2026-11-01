"""PrintPupper の段差・不整地歩行タスク。平地版 (env/joystick.py) との差分だけ。

設計の要点:

1. 地形は hfield 1枚で表現し、geom 名は平地版と同じ "floor" にする。
   pupper.xml の接触センサは geom2="floor" で引いているので、名前さえ保てば
   足の接地判定 (feet_air_time / clearance / slip) がそのまま動く。箱を並べた
   階段にするとこのセンサが拾えなくなるので hfield を選んでいる。
   warp / jax 両バックエンドで hfield 衝突が動くことは確認済み。

2. 高さ基準を「ワールド z 絶対値」から「足元の地面からの相対高さ」に変える
   (joystick.py の _ground_height フック経由)。平地版の base_height や
   feet_clearance をそのまま段差に持ち込むと、段に乗った瞬間にペナルティが
   入り「段を避けて平地だけ歩く」方策に収束する。ここが一番の罠。

3. 観測 (actor 38次元) は平地版と完全に同一。地形の情報は一切入らない。
   実機は IMU しか積んでおらず外受容センサが無いため、盲目歩行にするしかない。
   観測次元が同じなので、平地で学習した重みをそのまま初期値にできる
   (scripts/train.py --restore)。

4. difficulty は 0.0 (完全な平地) - 1.0 (最大段差 STEP_HEIGHT=3cm) の連続値。
   hfield の中身を書き換えるだけなので XML の作り直しは不要。平地の重みから
   difficulty を段階的に上げて fine-tune していく運用を想定している。
"""
from typing import Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict

from . import constants as consts
from .joystick import PupperJoystick, default_config


# ---------------------------------------------------------------- 地形生成
def _tile(nr: int, nc: int, cell: float, d: float, step_h: float,
          rough_h: float, rng: np.random.Generator):
    """タイル1枚の高さ [m]。外周は必ず平坦にして隣のタイルと連続させる。"""
    kind = rng.choice(("rough", "blocks", "stairs", "flat"), p=(0.3, 0.3, 0.3, 0.1))
    p = np.zeros((nr, nc), np.float32)

    if kind == "rough":
        # 不整地。セル単位の白色ノイズだと足半径 12mm に対して棘だらけになるので
        # 0.10m の粗い格子で作って最近傍アップサンプルする (IsaacGym 流)。
        k = max(1, int(round(0.10 / cell)))
        amp = rough_h * d * rng.uniform(0.5, 1.0)
        g = rng.uniform(0.0, amp, size=(nr // k + 1, nc // k + 1))
        p = np.repeat(np.repeat(g, k, 0), k, 1)[:nr, :nc].astype(np.float32)

    elif kind == "blocks":
        # 離散的な段差。0.30m 角のブロックを 0段/1段でばらまく。
        # 0-2 段にすると 0段と2段が隣り合って 2*step_h の崖ができる。
        # step_h=50mm では 98mm = 機体高さの 60% となり、登れる段ではなく壁。
        # 方策はそれを避けることしか学ばないので、公称値を超えさせない。
        k = max(1, int(round(0.30 / cell)))
        step = step_h * d * rng.uniform(0.6, 1.0)
        g = rng.integers(0, 2, size=(nr // k + 1, nc // k + 1)) * step
        p = np.repeat(np.repeat(g, k, 0), k, 1)[:nr, :nc].astype(np.float32)

    elif kind == "stairs":
        # ピラミッド階段。中心が高く外周が 0 なので、境界処理なしで連続する。
        # 登りと降りの両方が 1タイルに入るのが利点 (降りの方が難しい)。
        tread = max(1, int(round(0.15 / cell)))       # 踏面 0.15m
        riser = step_h * d * rng.uniform(0.5, 1.0)
        rr = np.arange(nr)[:, None] - nr / 2
        cc = np.arange(nc)[None, :] - nc / 2
        dist = np.maximum(np.abs(rr), np.abs(cc))
        lvl = np.clip((min(nr, nc) / 2 - dist) // tread, 0, 3)   # 最大3段
        p = (lvl * riser).astype(np.float32)

    b = max(1, int(round(consts.TERRAIN_BORDER / cell)))
    p[:b, :] = 0.0; p[-b:, :] = 0.0; p[:, :b] = 0.0; p[:, -b:] = 0.0
    # 上限は掛けない。高さは step_h / rough_h からしか生えないので、タイルの
    # 最大は階段の 3*step_h。縦スケールは set_hfield() が実行時に合わせる。
    return np.maximum(p, 0.0)


def heightmap(difficulty: float, seed: int = 0,
              step_h: float = consts.STEP_HEIGHT,
              rough_h: float = consts.TERRAIN_ROUGH) -> np.ndarray:
    """(N, N) の高さ [m]。行が y、列が x (MuJoCo hfield の並びに合わせる)。

    中央 TERRAIN_FLAT_R は平地にしてある。リセット直後に必ず足場があるように
    するためと、平地の方策がそのまま通用する領域を残して比較できるようにするため。
    """
    n, half = consts.TERRAIN_N, consts.TERRAIN_HALF
    cell = 2 * half / (n - 1)
    h = np.zeros((n, n), np.float32)
    d = float(np.clip(difficulty, 0.0, 1.0))
    if d <= 0.0:
        return h

    tile = consts.TERRAIN_TILE
    ntile = int(round(2 * half / tile))
    for ti in range(ntile):
        for tj in range(ntile):
            x0, y0 = -half + ti * tile, -half + tj * tile
            if max(abs(x0 + tile / 2), abs(y0 + tile / 2)) < consts.TERRAIN_FLAT_R:
                continue                                    # 中央は平地
            c0, c1 = int(round((x0 + half) / cell)), int(round((x0 + tile + half) / cell))
            r0, r1 = int(round((y0 + half) / cell)), int(round((y0 + tile + half) / cell))
            rng = np.random.default_rng(seed * 7919 + ti * 131 + tj)
            h[r0:r1, c0:c1] = _tile(r1 - r0, c1 - c0, cell, d,
                                    step_h, rough_h, rng)

    # 最重要の不変条件: 一歩で越える段差が公称値 step_h を超えないこと。
    # 破ると「登れない壁」が混ざり、方策はそれを避けることだけを学ぶ。
    # 地形の種類を足すときは必ずこの assert を通すこと。
    jump = max(np.abs(np.diff(h, axis=0)).max(), np.abs(np.diff(h, axis=1)).max())
    assert jump <= max(step_h, rough_h) * d + 1e-6, (
        f"最大段差 {jump*1000:.1f}mm が公称 {max(step_h, rough_h)*d*1000:.1f}mm を超えている")
    return h


def set_hfield(mj_model: mujoco.MjModel, hm: np.ndarray) -> None:
    """高さ場 hm [m] を hfield へ書き込む。縦スケールも hm に合わせる。

    MuJoCo は hfield_data (0-1) * hfield_size[2] を高さとして解釈する。size[2] は
    XML に TERRAIN_ZMAX で書いてあるが、ここで地形の最大高さへ上書きするので、
    XML の値は上限にならない。「階段 6段 x 40mm = 240mm」のように既定値を超える
    地形も XML を作り直さずに置ける。hfield_data は float32 なので、スケールを
    上げても分解能は zmax/2^23 (0.5m でも 6e-8 m) で、段差の形には影響しない。

    書き込みは MjModel を mjx へ転送する前に済ませること (hfield_size は
    mjx/warp 側では定数として焼かれる)。
    """
    zmax = max(float(hm.max()), 1e-3)          # 完全な平地でも 0 除算しない
    mj_model.hfield_size[0, 2] = zmax
    mj_model.hfield_data[:] = (hm / zmax).ravel()


# ---------------------------------------------------------------- 設定
def terrain_config(step_height: float = consts.STEP_HEIGHT) -> config_dict.ConfigDict:
    """平地版の設定から、段差で必要な箇所だけ緩めたもの。

    緩める理由はどれも同じで、平地版の正則化は「低く・水平に・すり足で」歩く
    ことを強制しており、それは段差を登れない歩容そのものだから。
    """
    cfg = default_config()
    cfg.terrain = config_dict.create(
        difficulty=0.5, seed=0,
        step_height=step_height,
        rough_height=consts.TERRAIN_ROUGH,
        spawn_radius=consts.TERRAIN_SPAWN_R,
    )

    s = cfg.reward_config.scales
    s.orientation = -2.0        # -5.0 から。段では胴体がピッチせざるを得ない
    s.lin_vel_z = -0.5          # -1.0 から。段の昇降で胴体は上下する
    s.feet_slip = -0.05         # -0.15 から。段の角で足を擦るのは避けられない
    # base_height は joystick.py 側で地面相対になったので重みは据え置きでよい。

    # 足上げ目標を「段差を確実に越える高さ」に上げる。位相クロックが指示する
    # rz プロファイルの頂点なので、ここが実質の最大跨ぎ高さになる。
    rc = cfg.reward_config
    rc.max_foot_height = consts.SWING_HEIGHT + step_height

    # feet_phase の sigma は平地版の 0.01 のまま触らない。締めると到達不能な
    # 目標の指数関数の裾に入り、「足を上げた場合」と「上げない場合」の報酬差
    # そのものが潰れる (実測: sigma 0.0026 で差 0.016、0.01 で差 0.043)。
    # get_rz の目標プロファイルは立脚期に 0 で留まる区間が無く、トロット位相を
    # 入れると 4本同時に浮いた姿勢 (飛翔期) を要求する。flight ペナルティと
    # 矛盾するので、この項は歩容の位相を揃える役にしか使えない。
    #
    # 段差越えに必要な「高く上げろ」という圧力は feet_clearance が与える
    # (片側罰: 遊脚最高到達点が目標に届いていない分だけを罰し、上げすぎは罰さない)。
    #
    # 重み換算: joystick.py でこの項を first_contact (接地の瞬間のみ) から
    # 遊脚窓の毎ステップ評価に変えた。発火頻度は
    #   旧 4脚 / 歩容周期 = 0.12〜0.22 発火/step
    #   新 4脚 x 遊脚窓 0.4       = 1.6  発火/step
    # で約 10 倍。罰の大きさも「最終到達点の不足」から「遊脚中の走査最大の不足」に
    # 変わり 1.2 倍程度増える。旧 -60 と同等の圧力なら -5 前後、その約2倍を狙って
    # -10 とする。強すぎると速度追従を潰すので、学習中は progress の 速度 欄と
    # 遊脚 欄の両方を見ること (遊脚が伸びずに速度だけ落ちていたら失敗)。
    # 実測で確定した値 (s30c -> d30b, 30mm地形 200M steps):
    #   疎 -60 : 遊脚 中央 23.5mm / p90 39.3mm / 越えられる段差 30mm
    #   密 -10 : 遊脚 変化なし (21M steps で不動)。密にするだけでは足りない
    #   密 -40 : 遊脚 中央 53.7mm / p90 76.0mm / 越えられる段差 40mm  <- これ
    # 密 -40 は 1ステップ報酬が負になるので「早く転んで支払いを止める」縮退の
    # 危険がある (env/balance.py のコメント参照)。実際には ep_len 970-1000 を
    # 維持したが、重みをこれ以上上げるなら ep_len を必ず監視すること。
    s.feet_clearance = -40.0

    # 接触バッファ。hfield は plane より接触が増えるが、実測すると
    # 512env / 段差50mm / ランダム行動の最悪時でも 2.62 接触/env しか出ない
    # (足球 4 + 胴体箱。hfield-box は MULTICCD 非対応で 1 接触に潰れる)。
    # naconmax はバッチ全体の共有バッファなので、ここを大きく取りすぎると
    # warp が XLA プールの外でメモリを食って OOM する。3倍の余裕で 8/env。
    cfg.naconmax = 8 * 4096
    cfg.njmax = 160
    return cfg


# ---------------------------------------------------------------- 環境
class PupperTerrain(PupperJoystick):
    """段差・不整地での速度コマンド追従歩行。観測は平地版と同一 (盲目)。"""

    XML = consts.TERRAIN_XML

    def __init__(
        self,
        config: config_dict.ConfigDict = terrain_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    # ---------- モデル ----------
    def _edit_model(self, mj_model: mujoco.MjModel) -> None:
        tc = self._config.terrain
        hm = heightmap(tc.difficulty, tc.seed, tc.step_height, tc.rough_height)
        set_hfield(mj_model, hm)
        self._t_map = jp.asarray(hm)
        self._t_n = consts.TERRAIN_N
        self._t_cell = 2 * consts.TERRAIN_HALF / (consts.TERRAIN_N - 1)
        self._t_max = float(hm.max())

    # ---------- 地形 ----------
    def _grid(self, xy: jax.Array):
        """xy -> 格子の左下インデックスと格子内の位置 (0-1)。範囲外は端に丸める。"""
        f = (xy + consts.TERRAIN_HALF) / self._t_cell
        f = jp.clip(f, 0.0, self._t_n - 1.001)
        i = f.astype(jp.int32)
        return i, f - i

    def _ground_height(self, xy: jax.Array) -> jax.Array:
        """双線形補間。MuJoCo の実際の三角形分割とは段の際で最大 1セル分ずれるが、
        報酬の基準としては十分 (ずれは格子間隔 0.025m のスケール)。"""
        i, t = self._grid(xy)
        c, r = i[..., 0], i[..., 1]
        tx, ty = t[..., 0], t[..., 1]
        h = self._t_map
        h00, h01 = h[r, c], h[r, c + 1]
        h10, h11 = h[r + 1, c], h[r + 1, c + 1]
        return (h00 * (1 - tx) + h01 * tx) * (1 - ty) + \
               (h10 * (1 - tx) + h11 * tx) * ty

    def _ground_height_max(self, xy: jax.Array) -> jax.Array:
        """近傍4点の最大値。スポーン時に地面へめり込ませないための安全側の評価。"""
        i, _ = self._grid(xy)
        c, r = i[..., 0], i[..., 1]
        h = self._t_map
        return jp.maximum(jp.maximum(h[r, c], h[r, c + 1]),
                          jp.maximum(h[r + 1, c], h[r + 1, c + 1]))

    def _spawn(self, rng: jax.Array, qpos: jax.Array) -> jax.Array:
        """地形上のランダムな位置に、地面高さぶん持ち上げて置く。

        難易度は hfield 全体で共有 (MJX では env ごとに hfield を変えられない)
        ため、多様性はスポーン位置のばらつきで確保する。中央 TERRAIN_FLAT_R は
        平地なので、簡単な初期条件も一定割合で必ず混ざる。
        """
        r = self._config.terrain.spawn_radius
        xy = jax.random.uniform(rng, (2,), minval=-r, maxval=r)
        return qpos.at[0:2].add(xy).at[2].add(self._ground_height_max(xy) + 0.005)

    def _get_termination(self, data, info) -> jax.Array:
        # 地形の外へ出ると足場が無く落下し続ける (姿勢は崩れないので fall では
        # 検知できない)。明示的に打ち切る。
        out = jp.max(jp.abs(data.xpos[self._torso_body_id, :2])) > consts.TERRAIN_HALF - 1.0
        return super()._get_termination(data, info) | out
