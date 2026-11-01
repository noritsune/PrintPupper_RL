# PrintPupper_RL

3D プリント製の四脚ロボット [PrintPupper](https://github.com/noritsune/PrintPupper)（v0.2、サーボ 12 個、膝は 4 節リンク駆動）を、MuJoCo MJX と brax の PPO で強化学習するリポジトリです。学習した方策は numpy だけで推論できる形に書き出し、IMU だけを積んだ実機（Raspberry Pi 3B）で動かします。

- 観測は 38 次元（IMU 9 + 速度指令 3 + 行動履歴 24 + 歩容の位相 2）、行動は 12 関節の目標角
- 方策は 1 つのネットワークで、段差を越える歩行と、転倒からの起き上がりの両方をこなします

## できること

成績はすべてシミュレーション上の値です。実機での成績はまだ測っていません。

| スキル | 内容 | 主な成績 |
|---|---|---|
| 統合方策（u60_stand、推奨） | 歩行と転倒復帰を 1 つの方策で切り替えなしに行う | 復帰 87.9%、段差 30mm 91.7% / 40mm 79.2% |
| 段差歩行 | ランダムな段差地形を速度指令どおりに歩く | 統合方策の歩行部分の教師として使用 |
| 転倒復帰 | 仰向け・横倒し・伏せから起き上がる | 成功率 100%、仰向けからの復帰時間の中央値 1.28 秒 |
| 対角 2 脚バランス | 2 本の脚だけで立ち続ける | 94.7% が 300 秒立ち続ける |

統合方策の成績は膝ホーン 60mm の機体での値、歩容は 1.2Hz に固定しています。

## 動作環境

次の環境で動かしています。

- Windows 11 + WSL2（Ubuntu 24.04）。ビューアは WSLg の画面に出ます
- NVIDIA GPU（RTX 3060 Ti 8GB）。学習は CUDA GPU が前提です
- Python 3.12

主要パッケージのバージョンは次のとおりです。

| パッケージ | バージョン |
|---|---|
| jax / jaxlib（CUDA 13） | 0.11.2 |
| mujoco / mujoco-mjx / mujoco-warp | 3.13.0 |
| playground（mujoco_playground） | 0.2.0 |
| brax | 0.14.2 |
| flax / optax | 0.12.9 / 0.2.8 |
| onnx / onnxruntime / torch | 1.23.0 / 1.30.0 / 2.14.0（ONNX 書き出しのみ） |

brax 0.14.2 と jax 0.11 を組み合わせるための互換処理を [env/_compat.py](env/_compat.py) に入れています。

## セットアップ

requirements ファイルは用意していません。上の表のバージョンに合わせて入れてください。

```bash
git clone https://github.com/noritsune/PrintPupper_RL.git
cd PrintPupper_RL
python3 -m venv .venv && source .venv/bin/activate
pip install "jax[cuda13]==0.11.2" mujoco==3.13.0 mujoco-mjx==3.13.0 mujoco-warp==3.13.0 \
  playground==0.2.0 brax==0.14.2 flax==0.12.9 optax==0.2.8 \
  trimesh etils mediapy matplotlib tensorboard
pip install onnx onnxruntime torch   # ONNX に書き出す場合だけ
```

以下のコマンドは、すべてリポジトリのルートで venv を有効にした状態で実行します。

## 使い方

学習結果は `out/train/<tag>/` に保存され、以降のコマンドの `<ckpt>` にはそこの `ckpt/final.pkl` を渡します。学習しなくても、同梱の最新の統合方策 [checkpoints/u60_stand.pkl](checkpoints/u60_stand.pkl) を `<ckpt>` に渡せば評価やビューアを試せます。

### 1. ロボットモデルを作る

```bash
python scripts/build_mjcf.py       # model/*.xml を生成する（生成済みのものを同梱）
python scripts/validate_model.py   # 物理パラメータの検証
```

`model/` には用途別に 4 つのモデルができます。平地用 `pupper.xml`、段差地形用 `pupper_terrain.xml`、転倒復帰用（脚にも衝突形状あり）`pupper_recover.xml`、統合方策用 `pupper_unified.xml` です。

### 2. 学習する

`--env` で課題を選びます（`flat` / `terrain` / `balance` / `recover` / `unified` / `climb`）。

```bash
# 平地歩行
python scripts/train.py --env flat --tag walk

# 段差歩行（平地歩行の方策から続ける）
python scripts/train.py --env terrain --difficulty 1.0 --step-height 0.03 \
  --restore out/train/walk/ckpt/final.pkl --tag step

# 転倒復帰（歩行の方策から、価値関数だけ作り直して始める）
python scripts/train.py --env recover --restore out/train/step/ckpt/final.pkl \
  --fresh-critic --num-envs 4096 --contacts-per-env 16 --num-timesteps 200000000 --tag rec

# 対角 2 脚バランス
python scripts/train.py --env balance --num-envs 4096 --num-timesteps 300000000 --tag bal
```

学習の進み具合は TensorBoard で見られます。

```bash
tensorboard --logdir out/train
```

統合方策は、歩行の方策と復帰の方策を 1 つに蒸留してから PPO で仕上げ、もう一度蒸留して作ります。

```bash
# 1. 歩行と復帰を 1 つのネットワークに蒸留
python scripts/distill.py --walk out/train/step/ckpt/final.pkl --recover out/train/rec/ckpt/final.pkl \
  --handoff-gate 1.0 --iters 20000 --out out/train/u_distill/ckpt/final.pkl

# 2. 探索の幅を絞ってから、統合環境で PPO
python scripts/set_policy_std.py --in out/train/u_distill/ckpt/final.pkl \
  --out out/train/u_distill_s/ckpt/final.pkl --std 0.2
python scripts/train.py --env unified --restore out/train/u_distill_s/ckpt/final.pkl --fresh-critic \
  --lr 1e-4 --entropy 0.001 --num-envs 2048 --contacts-per-env 32 --num-eval-envs 64 \
  --num-timesteps 200000000 --tag u

# 3. 静止状態からの歩き出しを含めて再蒸留
python scripts/distill.py --walk out/train/step/ckpt/final.pkl --recover out/train/u/ckpt/final.pkl \
  --handoff-gate 0.9 --out out/train/u_stand/ckpt/final.pkl \
  --iters 20000 --still-steps 300 --dagger 3 --dagger-iters 8000
```

GPU メモリが 8GB の場合、統合環境は `--num-envs 4096` だとメモリ不足になります。2048 に下げてください。

各スクリプトの全オプションは `--help` で確認できます。

### 3. 評価する

```bash
python scripts/step_test.py --ckpt <ckpt> --gait-freq 1.2 1.2      # 段差の成功率
python scripts/recover_test.py --ckpt <ckpt> --episodes 256        # 転倒復帰の成功率
python scripts/balance_test.py --ckpt <ckpt>                       # バランスの保持時間
python scripts/cmd_test.py --ckpt <ckpt>                           # 速度指令への追従
python scripts/check_deploy.py --ckpt <ckpt> --obs-history 0       # 実機用の推論と学習側の一致
```

- `step_test.py` の `--gait-freq` は、学習時と同じ値を必ず明示してください。違う値で測ると成功率が大きく変わります
- `check_deploy.py` の `--obs-history` は既定値が 10 です。このリポジトリの 38 次元の方策では `0` を指定します

### 4. ビューアで見る

WSLg の画面に MuJoCo のビューアが開き、マウスで視点を動かせます。

```bash
python scripts/view.py                                    # 基準姿勢で静止
python scripts/view.py --gait                             # スクリプト歩容を再生
python scripts/view.py --step 30 --policy <ckpt>          # 30mm の段差 1 つ
python scripts/view.py --stairs 30 --policy <ckpt>        # 30mm の階段
python scripts/view.py --recover --roll 180 --policy <ckpt>           # 仰向けから復帰
python scripts/view.py --unified --terrain --policy <ckpt>            # 統合方策をランダム地形で
python scripts/view.py --unified --terrain --policy pupper_policy.onnx  # 同梱の ONNX で
```

W/S/A/D と ←/→ で速度指令を変え、P でロボットを置き直し、T でカメラ追従を切り替えます。

### 5. 実機用に書き出す

```bash
python scripts/export_npz.py --ckpt <ckpt> --out deploy/policy_u_stand.npz   # 実機用（推奨）
python scripts/export_onnx.py --ckpt <ckpt> --out pupper_policy.onnx         # ONNX
```

`--ckpt` の既定値は学習済みの特定の run を指しているので、必ず指定してください。

### 6. 実機で動かす

実機側のコードは PrintPupper の [feat/rl-policy-walk](https://github.com/noritsune/PrintPupper/tree/feat/rl-policy-walk) ブランチの `src/rl/` にあります。**実機で動かせるのはこのブランチだけです。** PrintPupper の main ブランチには、方策で歩くためのコードが入っていません。

1. Windows で PrintPupper の `feat/rl-policy-walk` ブランチをチェックアウトし、`.\sync_to_robot.ps1 -PullPolicy` を実行します。WSL の `deploy/policy_u_stand.npz` が `src/rl/` にコピーされ、ロボットに転送されます
2. ロボット上で `robot.service` を起動します
3. ゲームパッドのボタン 12 で方策歩行（青 LED が点灯）、ボタン 11 でルールベース歩行に切り替えます。L1 で開始と停止です

詳しい手順と、サーボを動かさない確認モードは、同じブランチの [src/rl/README.md](https://github.com/noritsune/PrintPupper/blob/feat/rl-policy-walk/src/rl/README.md) にあります。

## 同梱の学習済み方策

| ファイル | 中身 |
|---|---|
最新は統合方策 u60_stand（膝ホーン 60mm）です。同じ方策を 3 つの形式で置いています。

| ファイル | 形式と用途 |
|---|---|
| [checkpoints/u60_stand.pkl](checkpoints/u60_stand.pkl) | 学習時のチェックポイント。評価・ビューア・追加学習（`--restore`）に使います |
| [deploy/policy_u_stand.npz](deploy/policy_u_stand.npz) | 実機用の重み。実機ではこれを使います |
| [pupper_policy.onnx](pupper_policy.onnx) | ONNX。入力 `obs[38]`、出力 `action[12]`（tanh を掛ける前の行動平均）。重みは `pupper_policy.onnx.data` に分かれています |

npz は [deploy/numpy_policy.py](deploy/numpy_policy.py) で、numpy だけで推論できます。ONNX は onnxruntime などで推論し、出力に tanh を掛けると実機用 npz と同じ行動になります。

[deploy/policy_u_stand_l50.npz](deploy/policy_u_stand_l50.npz) は、膝ホーン 50mm の機体向けに学習した旧版 u_stand の実機用の重みです。

## ディレクトリ構成

```
env/        学習環境
  joystick.py   平地歩行（観測と報酬の基本形）
  terrain.py    段差地形
  recover.py    転倒復帰
  unified.py    歩行と転倒復帰の統合
  balance.py    対角 2 脚バランス
  climb.py      段差の準静的な登坂
  linkage.py    膝の 4 節リンク（JAX 実装）
  deploy.py     実機用の観測の組み立て、ゲート、サーボ角変換
  constants.py  基準姿勢、指令範囲、地形などの定数
  randomize.py  ドメインランダム化
scripts/    学習・評価・可視化・書き出しのスクリプト
checkpoints/  最新の学習済みチェックポイント
model/      MJCF、STL メッシュ、URDF
deploy/     実機用の推論コード（numpy_policy.py, imu.py）と方策の重み
backup_L3_50/  膝ホーン 50mm 版のリンク計算
OUTPUTS.md  開発中の実験記録（学習の経過、各方策の評価、試したこと）
```

`OUTPUTS.md` は開発時の記録です。参照している `out/` 以下の動画・チェックポイント・ログはリポジトリに含まれていません。

## このロボット固有の注意点

- **膝の 4 節リンク**: 膝のサーボは股元にあり、リンクを介して下腿を動かします。関節角の上下限だけでは実機の可動域を表せないので、リンクで実現できない姿勢を報酬で罰しています（`linkage_infeasible`）。この罰を外すと、実機で取れない姿勢を覚えます
- **膝ホーンの長さ**: 実機 v0.2 は 60mm で、[env/linkage.py](env/linkage.py) と [scripts/knee_linkage.py](scripts/knee_linkage.py) も 60mm です。[env/deploy.py](env/deploy.py) のサーボ角変換は 50mm のままなので、実機では PrintPupper 側の `HardwareInterface` でサーボ角に変換します
- **歩容の周波数**: 統合方策は 1.2Hz で評価し、実機でも 1.2Hz で使います
- **IMU の取り付け**: 傾きは 1.7° 以内に収めてください（[deploy/imu.py](deploy/imu.py)）
- **脚の並び**: このリポジトリは lf, rf, lh, rh、PrintPupper は FR, FL, BR, BL の順です
- **活性化関数**: 隠れ層は silu です。tanh と取り違えてもエラーにならず、行動だけがずれます

## ライセンス

[MIT License](LICENSE)

`model/meshes/` の STL と `model/robot_work.urdf` は [r9kawai/PrintPupper_ROS](https://github.com/r9kawai/PrintPupper_ROS)（MIT License）に由来します。著作権表示は [model/LICENSE](model/LICENSE) にあります。
