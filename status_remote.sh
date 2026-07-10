cd /workspace/mfm
if pgrep -f run_gonogo.py >/dev/null 2>&1; then
  echo "状态: ▶ 训练进行中 (PID $(pgrep -f run_gonogo.py | head -1))"
else
  echo "状态: ■ 没有训练在跑（已结束或还没开始）"
fi
echo "GPU 空闲: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null)"
echo "------ out.log 末尾 ------"
tail -n 28 out.log 2>/dev/null
if grep -q VERDICT out.log 2>/dev/null; then
  echo "------ 终判 ------"
  sed -n '/===========/,$p' out.log | tail -n 12
fi
