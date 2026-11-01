"""PrintPupper 段差乗り越え (準静的登坂) タスク。

やること: 平地側に立った状態から、正面の 1 段 (既定 60mm) の上に機体ごと乗る。
歩行方策 (env/terrain.py) が 40mm で頭打ちになっている理由は機構ではないので、
**トロット歩容の枠を外した別スキル**として立てる。env/balance.py / env/recover.py
と同じ位置づけで、最後に scripts/distill.py で歩行・復帰と 1 本に統合する。

## なぜ歩行方策の延長では 60mm が出ないのか (実測・幾何から)

  - 足上げの上限は `action_scale` のクリップで決まる。歩行版の 0.25 では
    足先は股関節基準で 34.2mm しか上がらない。4節リンクと関節リミットが
    許す上限は 183.7mm、足先が股関節の 50-100mm 前方にある姿勢でも
    109-120mm なので、**機構は 60mm の段に対して十分な余裕がある**。
  - トルクも余っている。1 脚で全体重を支え、胴体高 80mm まで沈み込んだ
    最悪姿勢でも膝 1.58Nm (DS3235 stall 3.0Nm の 53%)。歩行中の実測は
    平均 0.30Nm・飽和 0.0%。
  - 静的安定余裕も足りる。基準姿勢で前脚を上げると余裕 +3.4mm (そのまま
    持ち上がる)、後脚側は -3.4mm だが重心を 12-19mm 動かせば +15mm 取れる。
  - それでも登れないのは報酬の形のせい。歩行版は
      `feet_phase=1.5` (sigma 0.01) が 4 脚全部を位相クロックの高さ
      プロファイルに貼り付け、duty 0.6 のトロットを強制する
      -> **3脚支持で胴体を振る動作は「学習されていない」のではなく罰される**
      `tracking_lin_vel=2.0` が毎ステップ効く
      -> 段の前で数秒止まる準静的登坂は、報酬順位で「突っ込んで失敗する
         トロット」に負ける
    ここを外さない限り、学習を足しても 50mm が天井になる。

## この環境で変えたこと

1. 歩容クロックを止める (`gait_freq=0`, `feet_phase=0`)。位相は定数になるので
   観測の (cos, sin) は (1, 0) 固定。**観測は歩行版と同一の 38 次元**のままで、
   蒸留と `--restore` の互換を保つ (env/balance.py と同じ作り)。
2. 速度追従を外し、「段に乗れたか」を直接評価する報酬に置き換える
   (`progress` / `climb` / `feet_on_ledge` / `body_over`)。
3. **静的安定余裕 `support` を報酬に入れる**。接地している足の支持多角形に
   対する重心の符号付き距離で、これがトロットでは表現できなかった
   「片足を上げきって保持し、重心を送ってから次の足」の直接の駆動力になる。
4. 関節別の `action_scale`。股関節 0.25 据え置き (横歩きの上限は足りている)、
   大腿・下腿だけ 0.60 -> 足上げ 34.2mm から 102.9mm へ。60mm の段に対して
   43mm の余裕。足を上げたときの前方到達も 40mm 上げで 99.2mm あり、後脚を
   段に乗せるのに必要な胴体前進 (+37mm) に届く。**実機に出すときは
   env/deploy.py と env/constants.py の写しを同時に直すこと**
   (scripts/check_deploy.py が数値照合する)。
5. **胴体接地で終了しない**。登坂中は段の縁に腹が当たりうる。歩行版の
   終了条件のままだと、当たった瞬間にエピソードが切れて登る動作そのものが
   探索から消える。罰 (`trunk_contact`) に格下げする。
6. 姿勢の罰をロールとピッチに分ける。段を登る機体は必ずピッチするので、
   歩行版の `orientation=-5.0` (ロールとピッチの区別なし) は登坂を罰する。

## 地形

x >= 0 が段の上。高さは y の帯ごとに変える (scripts/step_test.py の StepBench と
同じ手)。MJX は env ごとに hfield を変えられないので、**高さの多様性は
スポーンする帯で作る**。30-70mm の帯を並べてあるので、低い帯が学習初期の
足場になり、そのままカリキュラムになる。帯をまたぐと段の高さが曖昧になるので、
帯から出たら打ち切る。

## 使い方

```bash
# 歩行の重みから始める (観測 38 次元が同一なのでそのまま入る)。
# action_scale が違うので初期の挙動は壊れるが、env/recover.py (0.25 -> 1.6) で
# 同じことをして成立している。探索を絞るのが肝。
python scripts/train.py --env climb --restore out/train/e40/ckpt/final.pkl \
    --lr 1e-4 --entropy 0.001 --tag c60

# 成績を測る (帯ごとの登坂成功率 / どこで詰まるか / 実機成立性)
python scripts/climb_test.py --ckpt out/train/c60/ckpt/final.pkl --trials 32
```
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as pg_math
from mujoco_playground._src import gait as pg_gait
from mujoco_playground._src import mjx_env

from . import constants as consts
from .linkage import infeasibility
from .terrain import PupperTerrain, set_hfield, terrain_config

# 支持多角形の周回順。胴体を上から見て反時計回りに一周する。
# LEGS = [lf, rf, lh, rh] なので lf(+x,+y) -> lh(-x,+y) -> rh(-x,-y) -> rf(+x,-y)。
# 脚が交差しない限りこの順序は接地している部分集合でも保たれるので、
# 凸包を計算せずに外周の辺を列挙できる。
CYCLE = (0, 2, 3, 1)


def support_margin(feet_xy: jax.Array, contact: jax.Array,
                   com_xy: jax.Array) -> jax.Array:
    """接地脚が作る支持多角形に対する重心の符号付き距離 [m]。正なら内側。

    接地 2 点のときは支持基底が線分になるので、辺 (a->b) と (b->a) の両方が
    外周として立ち、最小値 = -|線分までの距離| が返る (静的余裕は必ず負)。
    接地 0-1 点のときは多角形が作れないので固定値 -0.1 を返す。
    """
    p = feet_xy[jp.array(CYCLE)]           # (4, 2) 周回順の足先 xy
    c = contact[jp.array(CYCLE)]           # (4,)   周回順の接地フラグ
    best = jp.array(jp.inf)
    for a in range(4):
        for d in (1, 2, 3):
            b = (a + d) % 4
            # a から b へ周回方向に進む間に接地脚が無いときだけ、これは外周の辺。
            active = c[a] & c[b]
            for k in range(1, d):
                active &= ~c[(a + k) % 4]
            e = p[b] - p[a]
            n = jp.sqrt(jp.sum(jp.square(e))) + 1e-9
            # 反時計回りなので、多角形の内部は各有向辺の左側 = 外積が正。
            cross = e[0] * (com_xy[1] - p[a][1]) - e[1] * (com_xy[0] - p[a][0])
            best = jp.minimum(best, jp.where(active, cross / n, jp.inf))
    return jp.where(jp.isinf(best), -0.1, best)


def climb_config(step_height: float = 0.060) -> config_dict.ConfigDict:
    """段差乗り越えの設定。step_height は「狙う段の高さ」で、帯はこれを中心に置く。"""
    cfg = terrain_config(step_height=step_height)

    # --- 歩容クロックは「弱く」残す ---
    # **c60-c60h で分かったこと**: 完全に止める (gait_freq=[0,0]) と、位相の
    # 観測が定数 (1,0) に固定され、歩行方策の脚の動きを駆動していた入力が
    # 消える。restore しても歩行能力は実質引き継げず、方策はゼロから移動を
    # 再発明することになる。6 本の run すべてで「転がって前進する」に落ちた
    # のは、それがクロック無しで発見できる唯一の移動手段だったから
    # (段上の足が 1.1-2.1/4 から動かず、安定余裕も -60mm 前後で頭打ち)。
    #
    # かといって歩行版の feet_phase=1.5 (sigma 0.01) をそのまま使うと、
    # 4脚全部がクロックの高さプロファイルに貼り付けられ、3脚支持で胴体を
    # 振る動作が罰される (これが 40mm の壁の正体だった)。
    # 低周波のクロックを弱い重みで残し、「歩き方の下敷き」としてだけ使う。
    # 周波数を下げるのは重心移送の時間を稼ぐため (OUTPUTS.md の
    # 「効いた一手 その2」と同じ理由)。
    cfg.gait_freq = [0.6, 1.0]
    # 段の前で数秒静止するのが正解の課題なので、外乱は既定で切る。
    # 登坂を覚えてから --pert-kick で足す。
    cfg.pert_config.enable = False
    cfg.episode_length = 500          # 10 s。準静的な登坂は 4-6s かかる

    # 地形生成は使わない (段は _edit_model が直接書く)。
    cfg.terrain.difficulty = 0.0

    cfg.climb = config_dict.create(
        # 帯ごとの段差高さ [m]。狙いの高さを中心に低い側を厚くする。
        # 低い帯が学習初期の足場になり、そのままカリキュラムになる。
        heights=[step_height - 0.030, step_height - 0.020, step_height - 0.010,
                 step_height, step_height + 0.010],
        band=1.6,                 # 帯の幅 [m]
        ledge_x=0.0,              # 段の縁の x [m]
        # 段の縁からどれだけ手前に置くか [m]。
        # **c60s で分かったこと**: 群別に測ると、段上開始は段の +145mm まで
        # 前進できているのに、手前開始は -182mm でほとんど動いていなかった。
        # 全体平均 -39mm はこの 2 群の打ち消し合いで、それを見て登坂側の
        # 重みを 3 回動かしていた。止まっていたのは「段を登る」ではなく
        # 「段まで歩いて行く」ほう。
        # この環境は元々「段の正面に立った状態から登る」スキルとして定義した
        # もので、長距離の移動は歩行方策 (e40) の仕事。スポーン範囲が定義と
        # ずれていたので、守備範囲に合わせて段の近くへ寄せる。
        spawn_x=[-0.20, -0.12],
        # **c60-c60j で分かったこと**: 「段まで歩く」と「段を登る」を同時に
        # 学ばせると、前者で駐車・後退の局所解に落ちて後者に到達しない。
        # 9 本の run すべてで段上の足が 0.6-2.1/4 から動かず、前進を強めれば
        # 転がり、安定を強めれば後退する振動が続いた。欠けているのは
        # 「前脚を段に掛けてから後脚を持ち上げる」動作そのもの。
        # この割合のエピソードを、前脚がすでに段の上にある姿勢から始める。
        # 復帰課題で「実際に詰まった姿勢を初期姿勢に混ぜる」(rec8) のが効いた
        # のと同じ手で、探索が届かない領域を初期分布で直接与える。
        p_on_ledge=0.5,
        # **幾何の制約 (実測)**: 足は真上には上がらず、上げるほど前に出る。
        # action_scale 0.60 では、足を 40mm 上げたときの足先は股関節の
        # +99.2mm、60mm 上げたときは +99.7mm 前方まで届く。後脚の股関節は
        # 胴体の -100mm なので、後脚を段の上 (x > +20mm) に置けるのは
        #     x_trunk - 0.100 + 0.099 > 0.020  ->  x_trunk > +21mm
        # つまり**胴体が段の縁を 21mm 以上越えてから**。
        # 最初にここを -80..-20mm に置いていたが、それは後脚が段に届かない
        # 姿勢で、完成しようのない初期状態を渡していた。
        on_ledge_x=[0.04, 0.10],     # そのときの胴体 x (段の縁から先) [m]
        spawn_y=0.25,             # 帯の中心からの横ばらつき [m]
        spawn_yaw=0.20,           # 正面向きからのばらつき [rad]
        pose_jitter=0.05,         # 初期関節角のばらつき [rad]
        # 関節別の可動域 [hip, upper_leg, lower_leg]。
        # 0.25 -> 足上げ 34.2mm / 0.60 -> 102.9mm (4節リンクの上限 183.7mm 以内)。
        # 横歩きの上限 0.161 m/s は指令 0.15 に足りているので股関節は広げない。
        # 上げた分だけ 1 目盛が粗くなる (0.25 比で 2.4 倍) が、60mm の段には
        # 足上げそのものより**前方到達**が効き、そちらが律速だった
        # (上の「幾何の制約 (実測)」を参照)。
        action_scale=[0.25, 0.60, 0.60],
        # 前進系の報酬を正立しているときだけ有効にするゲート (upvector z)。
        # 0.9 = 傾き 26度で満額、0.7 = 46度で 0。60mm の段を登るときの
        # ピッチは実測 17-25度なので、正常な登坂はゲートを通る。
        # **c60 (115M) で分かったこと**: これが無いと「段に突っ込んで転がり
        # ながら前進する」のが最適解になる。移動 0.17m/生存 500step から
        # 移動 1.9m/生存 69step へ学習が進み、報酬は上がるのに登坂は 0% のまま
        # だった。progress の重み 30 は Δx 1.9m で +57 を生み、生存を全部
        # 失っても釣りが来る。
        upright_gate=[0.7, 0.9],
        support_target=0.015,     # 静的安定余裕の目標 [m]。これ以上は加点しない
        success_x=0.35,           # 機体長 0.29m ぶん完全に乗り越えた位置 [m]
        # 段の縁から先をこれだけ超えたら「段の上」と数える [m]。
        # **c60m で分かったこと**: ここを 20mm にすると、残りの到達距離を
        # 食い潰して後脚が一生判定線に届かない。
        #   胴体が落ち着く位置 +32mm / 後脚の股関節 -68mm
        #   40mm 上げた足先の前方限界 +82.7mm -> 届く最前点 +14.7mm
        #   判定に必要な位置 +20mm  -> 5.3mm 足りない
        # 後脚は実際には段の上に乗っているのに報酬が 1 点も入らなかった。
        # 足球の半径は 12mm なので、5mm 超えていれば十分に段上と言える。
        on_ledge_margin=0.005,
        # 足が「段の上にある」と数える足先高さの上限 [m]。遊脚の頂点は
        # 目標 70mm なので、その半分あたりに置くと立脚と遊脚の前後が入り、
        # 跳ね上がった足や転倒中の足は入らない。
        on_ledge_z=0.035,
    )

    # 重みの大前提 (env/balance.py と同じ): 立っている間の 1 ステップ報酬が
    # 必ず正であること。罰が生存報酬を上回ると「早く転んで支払いを止める」のが
    # 最適になる。
    #
    # torque_limit は c60 では探索初期の飽和を見越して -0.1 に落としていたが、
    # 実測ではトルク飽和が 4.3 -> 6.0 -> 7.4% と上がり続けた (力任せに段へ
    # 押し込む方向に寄っていた)。歩行版と同じ -1.0 に戻す。ゼロから回す場合は
    # 最初だけ --scale torque_limit=-0.1 で緩めること。
    # 歩行版 1.5 に対する比率。**c60o で分かったこと**: 0.5 は弱すぎた。
    # 胴体が段の +164mm 先まで進んでいるのに後脚が判定線 (+5mm) に入らず、
    # 足先が股関節の 200mm 以上後方にある = **前脚で引っ張って後脚を
    # 引きずっている**状態だった。クロックを弱めたのは 3脚支持を許すため
    # だったが、後脚を振り出す強制力まで失っていた。
    # 1.2 なら 4 本とも振り出しは要求しつつ、歩行版ほど姿勢を固定しない。
    _phase_scale = 1.2
    cfg.reward_config = config_dict.create(
        scales=config_dict.create(
            feet_phase=_phase_scale,
            # --- 目的そのもの ---
            # 重みは「10s 立って登り切る」が「2m 転がる」に必ず勝つように置く。
            # エピソード総和の見積り (dt=0.02, 500 step):
            #   きれいに登る : climb +12 / feet_on_ledge +24 / body_over +14
            #                  survival +20 / support +10 / progress +7  = 約 +87
            #   段の手前で静止: survival +20 / support +15               = 約 +35
            #   突っ込んで転ぶ: 前進系はゲートで消え、生存も失う         = 約 +6
            survival=2.0,         # 1.0 から。転ばずに居続けることの価値を上げる
            # 転がり前進を止めるのは重みではなく正立ゲートの仕事なので、
            # ゲートを入れた分ここは戻してよい。c60b (10 に絞った) では
            # 「段の手前で安全に立つ」が強すぎて前へ出なくなった。
            # **c60f で分かったこと**: 変位を報酬する項は、転ぶことで必ず
            # farm できる。正立ゲートは 46度を超える転倒しか止めないので、
            # 30度で前へ崩れる動きは progress を稼ぎ続けた (到達 x +791mm
            # なのに段上の足 1.3/4・安定余裕 -65mm = 接地せず滑っている)。
            # 変位系は「段の方を向かせる」だけの弱い誘導に落とし、点数の
            # 主役は接地を伴う終状態 (feet_on_ledge / all_feet) に移す。
            # こちらは接地判定を含むので転んでは絶対に稼げない。
            # 30 -> 10 -> 20 -> 3 -> 8 -> 20。正立ゲート付き。
            # **c60q で分かったこと**: approach (毎ステップ) を成功線まで
            # 延ばしても、段の縁 +7.4 と成功線 +30.0 の差 22.6 では、後脚を
            # 上げ続けて前へ出るリスクに見合わない (到達 x が -25〜-53mm で
            # 振れ続けた)。c60p で後脚の振り出しが定着し、正立ゲートも入って
            # いるので、変位側の重みを戻しても転がりには落ちにくいはず。
            progress=20.0,
            # **c60i で分かったこと**: progress は telescoping なので
            # 「1m 進んで +8」の一回払いにしかならない。survival や
            # feet_on_ledge は毎ステップ入るので、一回払いでは
            # 「その場で安全に立つ」(+20/episode) に勝てず、段に近づかないまま
            # 歩容だけが安定した (到達 x -112mm で 26M 停滞)。
            # 歩行版が連続的に前進するのは tracking_lin_vel が毎ステップ
            # 効く項だから。同じ役割を、段までの距離に対する毎ステップの
            # 誘導項で入れる。段に着いたら頭打ちなので farming はできない。
            approach=3.0,
            # 段を登れたこと自体。Δ地面高さ は対称クリップで telescoping する
            # ので総和は「最終高さ - 初期高さ」に等しく、上限は帯の 70mm。
            # farming できない項なのでゲートを掛けず、重みで主役に据える。
            climb=50.0,           # 200 から。60mm 登って +3 の誘導項に落とす
            # **c60b で分かったこと**: 線形だと 2/4 で頭打ちになる。
            # 2本乗せて正立していれば survival+support+feet_on_ledge で
            # 約 55/episode 稼げてしまい、3本目を出して転ぶリスクに見合わない
            # (実測: 段上の足 2.1/4 で 45M step 横ばい、その後 1.7 へ後退)。
            # 割合の2乗にして、2/4 の価値を半分にしつつ 4/4 を 4 倍にする。
            feet_on_ledge=8.0,    # (割合)^2 で評価する。2/4 -> 0.25, 4/4 -> 1.0
            # 登り切ったことへの完了ボーナス。4脚とも段の上 & 正立のときだけ
            # 発火する。「半分上がった姿勢」の局所解から引き剥がす主動力。
            all_feet=10.0,
            body_over=2.0,        # 正立ゲート付き
            # --- 準静的であること ---
            # トロットでは表現できなかった「片足を上げきって保持する」の駆動力。
            # **c60d で分かったこと**: 両側評価にすると「その場で4脚立ち」が
            # 最適解になる。4脚接地の余裕は +90mm で目標 15mm を振り切って
            # 満額、脚を1本上げると +3.4mm に落ちるので、静止が歩行より
            # 4.4 倍得になっていた (平地の帯でも前進せず 到達 x +136mm で停止、
            # ep_len は 500/500 に張り付き)。
            # 片側罰に変える: 支持が崩れている分だけを罰し、余裕が正の側は
            # 加点しない (env/joystick.py の feet_clearance と同じ形)。
            # **c60g で分かったこと**: -100 は強すぎる。この方策の歩行には
            # 飛翔期があり (安定余裕の平均 -32.8mm)、歩くだけで 500step あたり
            # -33 を払う。生存 +20 を上回るので「段に近づかない」が最適に
            # なった (到達 x -99mm まで後退)。飛翔を抑えたいだけで歩行を
            # 禁止したいわけではないので、誘導程度に緩める。
            support=-20.0,        # 支持基底から外れた分 [m] だけを罰する
            flight=-1.0,          # 4脚とも浮くのは準静的登坂では要らない
            # --- 姿勢 ---
            # 段を登る機体は必ずピッチする。歩行版の orientation=-5.0 (ロールと
            # ピッチの区別なし) をそのまま持ち込むと登坂そのものを罰する。
            roll=-5.0,
            pitch=-0.5,
            base_height=-1.0,     # 歩行版の -2.0 より弱く。登坂中は沈み込む
            trunk_contact=-2.0,   # 終了条件から格下げしたもの
            feet_slip=-0.05,
            # --- 正則化 / ハードウェア保護 ---
            torques=-0.002,
            torque_limit=-1.0,    # -0.1 から。歩行版と同じ値に戻す
            action_rate=-0.02,
            energy=-0.001,
            dof_pos_limits=-2.0,
            linkage_infeasible=-5.0,
            termination=-50.0,
        ),
        base_height_target=consts.STAND_HEIGHT,
        # 足上げ目標。action_scale 0.60 での足先上昇は 102.9mm あるので、
        # 歩行版の「SWING_HEIGHT + 段差」(95mm) も物理的には届く。ただし
        # 律速は足上げでなく重心移送側なので、目標は据え置く (run d40: 目標を
        # 95mm に上げたら足は高く上がったが越えられる段差は 40->30mm に悪化)。
        max_foot_height=0.070,
        feet_phase_sigma=0.01,
        torque_soft_limit_frac=0.5,
    )
    return cfg


class PupperClimb(PupperTerrain):
    """段差乗り越え (準静的登坂)。観測は歩行版と同一 (actor 38, 盲目)。"""

    def __init__(
        self,
        config: config_dict.ConfigDict = climb_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

    def _post_init(self) -> None:
        super()._post_init()
        cc = self._config.climb
        self._scale = jp.array(
            np.tile(np.asarray(cc.action_scale, np.float32), 4))      # (12,)

    # ---------- モデル ----------
    def _edit_model(self, mj_model: mujoco.MjModel) -> None:
        """x >= ledge_x を段の上にする。高さは y の帯ごとに変える。"""
        cc = self._config.climb
        n, half = consts.TERRAIN_N, consts.TERRAIN_HALF
        cell = 2 * half / (n - 1)
        hm = np.zeros((n, n), np.float32)          # [row=y, col=x]
        grid = -half + np.arange(n) * cell
        heights = np.asarray(cc.heights, np.float32)
        centers = (np.arange(len(heights)) - (len(heights) - 1) / 2) * cc.band
        for h, yc in zip(heights, centers):
            rows = (grid >= yc - cc.band / 2) & (grid < yc + cc.band / 2)
            hm[np.ix_(rows, grid >= cc.ledge_x)] = h
        set_hfield(mj_model, hm)
        self._t_map = jp.asarray(hm)
        self._t_n = n
        self._t_cell = cell
        self._t_max = float(hm.max())
        self._band_y = jp.asarray(centers, jp.float32)
        self._band_h = jp.asarray(heights, jp.float32)

    # ---------- 段の上に乗った足 ----------
    def _on_ledge(self, feet_xpos: jax.Array, contact: jax.Array) -> jax.Array:
        """段の上にある足のフラグ (4,)。

        **その瞬間の接地を要求してはいけない。** トロットで歩く機体は常に
        2 本しか接地していないので、接地を条件にすると 4本同時は静止時に
        しか起こらない。実際 c60-c60o の全 run で段上の足がきっかり 2.0/4 で
        止まり、10mm の段 (遊脚 34mm の歩行方策でも跨げる高さ) でさえ 4本
        乗らなかった。「段を登る」ではなく「段を登って四つ足で静止する」を
        要求していたことになる。

        接地の代わりに「足が段の上にあり、かつ足元の地面に十分近い」で見る。
        立脚でも遊脚の前後でも成立し、宙に投げ出された足や転がっている
        機体では成立しない。
        """
        del contact
        cc = self._config.climb
        z = (feet_xpos[:, 2] - consts.FOOT_RADIUS
             - self._ground_height(feet_xpos[:, :2]))
        return ((feet_xpos[:, 0] > cc.ledge_x + cc.on_ledge_margin)
                & (z < cc.on_ledge_z))

    # ---------- 可動域 ----------
    def _motor_targets(self, applied: jax.Array, info) -> jax.Array:
        # 関節別のスケール。歩行版のスカラー版と違うのはここだけ。
        return self._default_pose + applied * self._scale

    # ---------- reset ----------
    def _spawn(self, rng: jax.Array, qpos: jax.Array) -> jax.Array:
        # 置き直しは reset() で行うので、地形版のランダム配置は素通りさせる。
        del rng
        return qpos

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # 歩行版の reset で info の作り方と乱数の消費を共有し、そのあと
        # 「段の正面に、正面を向いて立っている」状態に置き直す。
        state = super().reset(rng)
        info = state.info
        cc = self._config.climb

        info["rng"], kb, kx, ky, kw, kq = jax.random.split(info["rng"], 6)
        i = jax.random.randint(kb, (), 0, len(cc.heights))
        band_y = self._band_y[i]
        step_h = self._band_h[i]

        info["rng"], k_mode, k_on = jax.random.split(info["rng"], 3)
        on_ledge = jax.random.bernoulli(k_mode, cc.p_on_ledge)
        x_walk = jax.random.uniform(
            kx, (), minval=cc.spawn_x[0], maxval=cc.spawn_x[1])
        x_on = jax.random.uniform(
            k_on, (), minval=cc.on_ledge_x[0], maxval=cc.on_ledge_x[1])
        x0 = cc.ledge_x + jp.where(on_ledge, x_on, x_walk)
        y0 = band_y + jax.random.uniform(
            ky, (), minval=-cc.spawn_y, maxval=cc.spawn_y)
        yaw = jax.random.uniform(
            kw, (), minval=-cc.spawn_yaw, maxval=cc.spawn_yaw)

        # 手前は平地 (地面高さ 0) なので keyframe の高さをそのまま使える。
        # 前脚を段に乗せて始める場合は、段の高さぶん持ち上げて置く
        # (前脚は段の上、後脚は下の地面の上に落ちる)。
        lift = jp.where(on_ledge, step_h, 0.0)
        qpos = self._init_q.at[0].set(x0).at[1].set(y0).at[2].add(0.002 + lift)
        qpos = qpos.at[3:7].set(pg_math.quat_mul(
            self._init_q[3:7],
            pg_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)))
        qpos = qpos.at[7:].set(self._default_pose + jax.random.uniform(
            kq, (12,), minval=-cc.pose_jitter, maxval=cc.pose_jitter))
        data = state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel),
                                  ctrl=self._default_pose)
        data = mjx.forward(self.mjx_model, data)

        info["on_ledge_start"] = on_ledge.astype(jp.float32)
        info["band_y"] = band_y
        info["step_h"] = step_h
        info["spawn_x"] = x0
        # 速度コマンドの枠は使わない。蒸留で「登坂モード」を選ぶ枠として
        # [1, 0, 0] を固定で入れておく (env/balance.py が対角ペアに使ったのと同じ手)。
        info["command"] = jp.array([1.0, 0.0, 0.0])
        info["phase"] = jp.zeros(())
        info["gait_freq"] = jp.zeros(())
        info["last_x"] = x0
        info["last_ground"] = jp.mean(
            self._ground_height(data.site_xpos[self._feet_site_id][..., :2]))
        info["ground_delta"] = jp.zeros(())

        metrics = dict(state.metrics)
        for k in ("clb/x", "clb/feet", "clb/margin", "clb/height",
                  "clb/success", "clb/step_h", "clb/on_start",
                  "clb/feet_on", "clb/succ_on", "clb/x_on", "clb/x_walk"):
            metrics[k] = jp.zeros(())

        obs = self._obs_reset(data, info)
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    # ---------- step ----------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        info, data = state.info, state.data
        cc = self._config.climb

        x = data.xpos[self._torso_body_id, 0]
        feet_xpos = data.site_xpos[self._feet_site_id]
        contact = data.sensordata[self._feet_contact_adr] > 0
        on_ledge = self._on_ledge(feet_xpos, contact)
        ground = jp.mean(self._ground_height(feet_xpos[..., :2]))
        upright = self._gravity(data)[-1] < -0.8

        state.metrics["clb/x"] = x - cc.ledge_x
        state.metrics["clb/feet"] = jp.sum(on_ledge.astype(jp.float32))
        state.metrics["clb/margin"] = support_margin(
            feet_xpos[:, :2], contact, data.subtree_com[self._torso_body_id, :2])
        state.metrics["clb/height"] = ground
        state.metrics["clb/success"] = (
            (x > cc.ledge_x + cc.success_x) & (jp.sum(on_ledge) == 4) & upright
        ).astype(jp.float32)
        state.metrics["clb/step_h"] = info["step_h"]
        state.metrics["clb/on_start"] = info["on_ledge_start"]
        # 群平均は mean(x * on) / mean(on) で取れる。段上開始グループだけの
        # 成績を見ないと、全体平均では「段上グループが 2.0 のまま」と
        # 「全体が少しずつ動いている」を区別できない。
        state.metrics["clb/feet_on"] = (
            state.metrics["clb/feet"] * info["on_ledge_start"])
        state.metrics["clb/succ_on"] = (
            state.metrics["clb/success"] * info["on_ledge_start"])
        # 到達 x も群別に出す。全体平均だけでは「段上開始が前に出ない」のか
        # 「手前開始が段に届いていない」のかが区別できない。
        state.metrics["clb/x_on"] = (
            state.metrics["clb/x"] * info["on_ledge_start"])
        state.metrics["clb/x_walk"] = (
            state.metrics["clb/x"] * (1.0 - info["on_ledge_start"]))
        # 親が入れる歩行用の指標は意味が変わるので、この課題の量に差し替える。
        v = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
        state.metrics["track/speed"] = jp.linalg.norm(v[:2])
        state.metrics["track/cmd_speed"] = jp.zeros(())
        state.metrics["track/lin_err"] = jp.zeros(())
        state.metrics["track/ang_err"] = jp.zeros(())

        # AutoReset は data/obs しか巻き戻さず info はそのまま流れてくるので、
        # 差分の基準を自分で戻す (env/balance.py の hold_steps と同じ手当て)。
        info["last_x"] = jp.where(state.done > 0, info["spawn_x"], info["last_x"])
        info["last_ground"] = jp.where(state.done > 0, 0.0, info["last_ground"])
        return state

    # ---------- 終了条件 ----------
    def _get_termination(self, data, info) -> jax.Array:
        cc = self._config.climb
        # 胴体接地では終了しない。登坂中は段の縁に腹が当たりうるので、
        # ここで切ると登る動作そのものが探索から消える (罰に格下げした)。
        fall = self._gravity(data)[-1] > -0.3
        y = data.xpos[self._torso_body_id, 1]
        # 帯から出ると段の高さが変わってしまい、何を登ったのか分からなくなる。
        band_out = jp.abs(y - info["band_y"]) > cc.band / 2 - 0.2
        back = data.xpos[self._torso_body_id, 0] < cc.ledge_x - 1.0
        return fall | band_out | back

    # ---------- コマンド ----------
    def sample_command(self, rng: jax.Array, prev: jax.Array) -> jax.Array:
        del rng
        return prev

    # ---------- 報酬 ----------
    def _get_reward(self, data, action, info, done, first_contact, contact):
        rc = self._config.reward_config
        cc = self._config.climb

        q = data.qpos[7:]
        gravity = self._gravity(data)
        feet_xpos = data.site_xpos[self._feet_site_id]
        contact_f = contact.astype(jp.float32)
        ground = jp.mean(self._ground_height(feet_xpos[..., :2]))

        # 前進。1 ステップ (20ms) で起きうる最大の移動で対称にクリップする。
        # 対称にするとエピソード総和が「最終 x - 初期 x」に telescoping し、
        # 段の縁で前後に揺れて稼ぐ縮退が消える (climb 報酬と同じ作り)。
        x = data.xpos[self._torso_body_id, 0]
        dx = jp.clip(x - info["last_x"], -0.01, 0.01)
        info["last_x"] = x

        on_ledge = self._on_ledge(feet_xpos, contact)

        margin = support_margin(
            feet_xpos[:, :2], contact,
            data.subtree_com[self._torso_body_id, :2])

        # 歩容クロックが指示する足先高さ。歩行版と同じ式だが、重みが 1/3 なので
        # 「この形に近いと少し得」程度の下敷きにしかならない。
        phi = 2 * jp.pi * info["phase"] + self._foot_phase_offset
        phi = (phi + jp.pi) % (2 * jp.pi) - jp.pi
        rz = pg_gait.get_rz(phi, swing_height=rc.max_foot_height)
        foot_z = (feet_xpos[:, 2] - consts.FOOT_RADIUS
                  - self._ground_height(feet_xpos[:, :2]))
        feet_phase = jp.exp(-jp.sum(jp.square(foot_z - rz)) / rc.feet_phase_sigma)

        # 正立ゲート。転がりながらの前進に報酬が付かないようにする。
        # 硬い閾値にすると境界で報酬が段差状に飛ぶので、帯で滑らかに落とす
        # (env/unified.py の可動域ゲートと同じ作り)。
        lo, hi = cc.upright_gate
        gate = jp.clip((-gravity[2] - lo) / (hi - lo), 0.0, 1.0)

        # 前進の誘導。**c60p で分かったこと**: 頭打ちを段の縁 (x = ledge_x) に
        # 置くと、段を越えた先へ進む誘導がゼロになる。後脚を振り出せるように
        # なっても到達 x が ±40mm で振れ続けた。頭打ちを成功線
        # (ledge_x + success_x) まで延ばし、登り切るところまで引き続ける。
        # 減衰長 0.25 -> 0.6。0.25 だと -300mm で +2.2/episode しか出ず、
        # 段から離れた位置では前進の駆動力がほぼゼロだった。
        behind = jp.clip(cc.ledge_x + cc.success_x - x, 0.0, None)
        approach = jp.exp(-behind / 0.6) * gate

        feet_vel = data.sensordata[
            (self._feet_linvel_adr[:, None] + np.arange(3)[None])
        ]
        over = jp.clip(jp.abs(data.actuator_force)
                       - self._stall * rc.torque_soft_limit_frac, 0.0, None)
        out_of_limits = jp.sum(
            jp.clip(self._soft_lowers - q, 0.0, None)
            + jp.clip(q - self._soft_uppers, 0.0, None)
        )
        base_h = data.xpos[self._torso_body_id, 2] - consts.FOOT_RADIUS - ground

        return {
            "survival": jp.ones(()),
            "feet_phase": feet_phase,
            "progress": dx / self.dt * gate,
            "approach": approach,
            "climb": info["ground_delta"] / self.dt,
            "feet_on_ledge": jp.square(jp.mean(on_ledge.astype(jp.float32))),
            "all_feet": (jp.sum(on_ledge) == 4).astype(jp.float32) * gate,
            "body_over": (x > cc.ledge_x).astype(jp.float32) * gate,
            # 片側罰。**閾値は 0 (支持基底の内か外か)** であって目標余裕では
            # ない。support_target (15mm) を閾値にすると、3脚支持の素の余裕
            # +3.4mm にも罰が付き、脚を上げること自体を罰してしまう。
            # 支持基底から出た分だけを罰することで:
            #   4脚接地 / 3脚支持で余裕あり -> 0     (静止も歩行も等しく無料)
            #   対角2脚 (-3.4mm)            -> -0.007/step (トロットは軽い罰)
            #   宙に浮く・転がる (-50mm 以上) -> -0.1/step  (これを潰したい)
            "support": jp.clip(-margin, 0.0, 0.05),
            "flight": (jp.sum(contact) == 0).astype(jp.float32),
            # ロールだけを強く罰する。ピッチは段を登る以上避けられない。
            "roll": jp.square(gravity[1]),
            "pitch": jp.square(gravity[0]),
            "base_height": jp.square(base_h - rc.base_height_target),
            "trunk_contact": (
                data.sensordata[self._trunk_contact_adr] > 0).astype(jp.float32),
            "feet_slip": jp.sum(jp.square(feet_vel[:, :2]) * contact_f[:, None]),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "torque_limit": jp.sum(jp.square(over)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            "dof_pos_limits": out_of_limits,
            "linkage_infeasible": infeasibility(q),
            "termination": done.astype(jp.float32),
        }
