#!/bin/bash
# 学習の進捗をまとめて確認する。
RUN=${1:-$(ls -td ~/pupper_rl/out/train/*/ 2>/dev/null | head -1)}
echo "=== $RUN ==="
echo
echo "--- 学習曲線 (直近10エントリ) ---"
grep -E "^  \[" ~/pupper_rl/out/train_*.log 2>/dev/null | tail -10
echo
echo "--- 書き出された動画 ---"
ls -1t "$RUN"video/*.mp4 2>/dev/null | head -8 | while read f; do
  printf "  %-34s %6.1f KB\n" "$(basename "$f")" "$(( $(stat -c%s "$f") / 1024 ))"
done
echo
echo "--- 最新の可視化 ---"
ls -la "$RUN"progress.png 2>/dev/null
echo
echo "Windows から見るには:"
echo "  動画  : explorer.exe \"\$(wslpath -w $RUN"video")\""
echo "  グラフ : explorer.exe \"\$(wslpath -w $RUN"progress.png")\""
echo "  TensorBoard: tensorboard --logdir ~/pupper_rl/out/train --bind_all  -> http://localhost:6006"
