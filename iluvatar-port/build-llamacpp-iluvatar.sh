#!/usr/bin/env bash
# build-iluvatar.sh — build llama.cpp (GGML_CUDA) for Iluvatar CoreX / BI-V150.
#
# Prerequisites (installed once by setup.sh in the repo):
#   /usr/local/bin/ivcc              CoreX clang CUDA-compiler wrapper
#   /usr/local/bin/corex_compat.h    API/codegen shims (force-included)
#   /usr/local/corex-4.4.0           CoreX toolchain (clang-18 "bi" target)
#
# Patches applied to the llama.cpp tree (idempotent, see patches/ in repo):
#   P1  numeric __CUDA_ARCH__ comparisons -> GGML_CUDA_ARCH_NUM (=500 via ivcc)
#       (keeps generic kernel paths; mma/dp4a PTX is rejected by the bi asm)
#   P2  __trap() -> no-op               (bi asm has no "trap")
#   P3  allreduce.cu -> built-in MUSA fallback branch (custom AR kernels
#       crash llc; nullptr pipeline = graceful generic AllReduce)
#   P4  mmvf.cu, lightning-indexer.cu -> stubs (kernels crash llc;
#       dispatchers fall back to cuBLAS gemv / CPU backend)
#   P5  softmax.cu: drop cooperative_groups/reduce.h (missing in CoreX)
set -e
COREX=/usr/local/corex-4.4.0
SRC=/root/llama-deploy/llama.cpp
BD=$SRC/build-iluvatar
STUBS=/root/llama-deploy/stubs
cd $SRC

# ---- P1/P2: macro + trap sed over ggml-cuda ------------------------------
if [ ! -f .ivcore_patched ]; then
  grep -rl '__CUDA_ARCH__' ggml/src/ggml-cuda --include='*.cu' --include='*.cuh' --include='*.cpp' --include='*.h' | while read -r f; do
    sed -i 's/__trap()/((void)0)/g; s/__CUDA_ARCH__ >=/GGML_CUDA_ARCH_NUM >=/g; s/__CUDA_ARCH__ ==/GGML_CUDA_ARCH_NUM ==/g; s/__CUDA_ARCH__ <=/GGML_CUDA_ARCH_NUM <=/g; s/__CUDA_ARCH__ >/GGML_CUDA_ARCH_NUM >/g; s/__CUDA_ARCH__ <\([^ =]\)/GGML_CUDA_ARCH_NUM <\1/g' "$f"
  done
  # ---- P5: softmax cooperative_groups/reduce.h ---------------------------
  sed -i 's|#include <cooperative_groups/reduce.h>|// iluvatar port: header missing in CoreX|' ggml/src/ggml-cuda/softmax.cu
  # ---- P3: allreduce -> MUSA fallback branch ------------------------------
  sed -i '3s/.*/#if 0 \/\/ iluvatar port: use fallback branch below/' ggml/src/ggml-cuda/allreduce.cu
  # ---- P4: stub TUs -------------------------------------------------------
  cp ggml/src/ggml-cuda/mmvf.cu ggml/src/ggml-cuda/mmvf.cu.orig
  cp ggml/src/ggml-cuda/lightning-indexer.cu ggml/src/ggml-cuda/lightning-indexer.cu.orig
  cp $STUBS/stub-mmvf.cu ggml/src/ggml-cuda/mmvf.cu
  cp $STUBS/stub-lightning.cu ggml/src/ggml-cuda/lightning-indexer.cu
  touch .ivcore_patched
fi

rm -rf $BD
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CUDACXX=/usr/local/bin/ivcc
export CUDA_PATH=$COREX CUDAToolkit_ROOT=$COREX
cmake -B $BD -S $SRC \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/bin/ivcc \
  -DCMAKE_CUDA_ARCHITECTURES=ivcore11 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DGGML_NATIVE=ON \
  > /root/llama-cuda-cfg.log 2>&1
echo CONFIGURE_OK
cmake --build $BD --target llama-server llama-cli -j 48 > /root/llama-cuda-build.log 2>&1
echo BUILD_OK
ls -la $BD/bin/llama-server
