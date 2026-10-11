#!/usr/bin/env bash
# serve-parallel.sh — 多实例并行 serving(Qwen3.8-27B,天数 BI-V150)
#
# 单卡 batch=1 的 decode 受显存带宽限制(~15 tok/s 上限);要聚合吞吐就必须
# (a)单卡开大 batch(-np),一步读一遍权重出 N 个 token;
# (b)每卡一个独立实例,显存带宽随卡数线性叠加。
# 4 卡 × -np 64 实测 167.4 tok/s 聚合(单流基线 9.2 → 18.2x)。
#
# 用法:
#   ./serve-parallel.sh <模型.gguf> [GPU列表] [起始端口] [每实例slot数] [每slot上下文]
# 示例:
#   ./serve-parallel.sh Qwen3.8-27B-UD-Q4_K_M.gguf "3,4,6,7" 8960 64 512
#
# 显存预算(BI-V150 32G):Q4_K_M 权重 ~16.2G + KV(64 slot × 512 ctx ≈ 32k token)
# + 计算缓冲 ≈ 30.2G,正好放得下;更大量化请按比例调小 -np 或 -c。

set -euo pipefail

MODEL=${1:?usage: serve-parallel.sh <model.gguf> [gpus] [base_port] [slots] [ctx_per_slot]}
GPUS=${2:-3,4,6,7}
BASE_PORT=${3:-8960}
SLOTS=${4:-64}
CTX=${5:-512}

BIN_DIR=$(cd "$(dirname "$0")/../build-iluvatar/bin" 2>/dev/null && pwd) \
  || BIN_DIR=""
SERVER="${BIN_DIR:-$(dirname "$(command -v llama-server 2>/dev/null || echo /usr/local/bin)")}/llama-server"
[ -x "$SERVER" ] || { echo "llama-server 不在 build-iluvatar/bin,也不在 PATH:请先跑 build-llamacpp-iluvatar.sh"; exit 1; }

TOTAL_CTX=$((SLOTS * CTX))
PIDS=()
IFS=',' read -ra GPU_ARR <<< "$GPUS"
for i in "${!GPU_ARR[@]}"; do
  GPU=${GPU_ARR[$i]}
  PORT=$((BASE_PORT + i))
  LOG=/tmp/llama-parallel-${PORT}.log
  echo "GPU $GPU -> port $PORT (-np $SLOTS, ctx/slot $CTX, total $TOTAL_CTX) log: $LOG"
  CUDA_VISIBLE_DEVICES=$GPU nohup setsid "$SERVER" \
    -m "$MODEL" -ngl 99 -t 8 --port "$PORT" -c "$TOTAL_CTX" -np "$SLOTS" \
    > "$LOG" 2>&1 < /dev/null &
  PIDS+=($!)
done

echo
echo "已启动 ${#GPU_ARR[@]} 个实例。健康检查:"
sleep 5
for i in "${!GPU_ARR[@]}"; do
  PORT=$((BASE_PORT + i))
  ( for t in $(seq 1 120); do
      curl -sf -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null && { echo "  port $PORT: ready"; exit 0; }
      sleep 5
    done; echo "  port $PORT: 未就绪(模型大,首次加载 ~4 分钟),查 $LOG" ) &
done
wait
echo "全部实例就绪。压测示例:"
echo "  seq 0 255 | xargs -P 256 -I{} curl -s http://127.0.0.1:${BASE_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' -d '{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64}'"
