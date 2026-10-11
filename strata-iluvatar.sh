#!/usr/bin/env bash
# strata-iluvatar.sh — patch + configure + build Strata for Iluvatar CoreX BI-V150.
#
# The port follows Strata's own STRATA_EXPERIMENTAL_SM60 compatibility mode
# (Pascal/Volta fallback paths), routed through a compile-time arch override:
#
#   P1  every `__CUDA_ARCH__ <cmp> N` comparison in src/, include/, bench/
#         -> `STRATA_CUDA_ARCH_NUM <cmp> N`, compiled with -DSTRATA_CUDA_ARCH_NUM=600
#       (comparisons only: `defined(__CUDA_ARCH__)` keeps its host/device meaning)
#         => >= 800 paths (mma.sync/cp.async/tf32) off, < 750 Volta paths on,
#            dp4a.hpp's sub-6.1 fallback on
#   P2  dp4a.hpp: the PTX-vmad fallback is replaced with plain C arithmetic on
#       the bi target (the bi assembler does not take `vmad.s32.s32.s32` byte
#       selectors) - still bit-exact, just slower
#
# Build: cmake with the ivcc wrapper as CUDA compiler, arch "60" so CMake's own
# checks accept it (the wrapper drops the -gencode flags and compiles for
# ivcore11). STRATA_EXPERIMENTAL_SM60=ON turns on the fallback code paths.
set -e

STRATA_SRC=$(cd "$(dirname "$0")" && pwd)
COREX=/usr/local/corex-4.4.0
BUILD=$STRATA_SRC/build-iluvatar

cd $STRATA_SRC

# ---- P1: arch comparison rewrite (idempotent via .iluvatar_patched) -------
if [ ! -f .iluvatar_patched ]; then
  grep -rl '__CUDA_ARCH__ *[<>=!]' src include bench 2>/dev/null | while read -r f; do
    sed -i 's/__CUDA_ARCH__ \([<>=!]\+\) /STRATA_CUDA_ARCH_NUM \1 /g' "$f"
  done
  # ---- P2: portable dp4a on the bi target --------------------------------
  # dp4a.hpp' PTX fallback (guarded by `STRATA_CUDA_ARCH_NUM < 610` after P1)
  # uses `vmad.s32.s32.s32` byte selectors that the bi assembler rejects.
  # Replace the asm body with plain integer arithmetic - identical semantics.
  python3 - <<'PY'
import re, pathlib
p = pathlib.Path("include/strata/kernels/dp4a.hpp")
s = p.read_text()
old = '''    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\\n\\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;'''
new = '''    // iluvatar: the bi assembler rejects the PTX vmad byte selectors, so the
    // same four byte products as plain arithmetic - bit-identical results.
    const signed char * pa = reinterpret_cast<const signed char *>(&a);
    const signed char * pb = reinterpret_cast<const signed char *>(&b);
    return c + int(pa[0]) * pb[0] + int(pa[1]) * pb[1] + int(pa[2]) * pb[2] + int(pa[3]) * pb[3];'''
if old in s:
    s = s.replace(old, new)
    p.write_text(s)
    print("dp4a.hpp: vmad -> plain arithmetic")
else:
    print("dp4a.hpp: vmad block not found (already patched?)")
PY
  touch .iluvatar_patched
fi

# ---- configure -------------------------------------------------------------
rm -rf $BUILD
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CUDACXX=/usr/local/bin/ivcc
export CUDA_PATH=$COREX CUDAToolkit_ROOT=$COREX
cmake -B $BUILD -S $STRATA_SRC \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/bin/ivcc \
  -DCMAKE_CUDA_ARCHITECTURES=ivcore11 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-L$COREX/lib64 -L/usr/local/iluvatar/lib64 -lcudart -lcublas -lcurand -lcuda -lixthunk -lixml" \
  -DCMAKE_CUDA_STANDARD_LIBRARIES="-L$COREX/lib64 -L/usr/local/iluvatar/lib64 -lcudart -lcublas -lcurand -lcuda -lixthunk -lixml" \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR=/root/llama-pinned \
  -DCMAKE_CXX_FLAGS="-DSTRATA_CUDA_ARCH_NUM=600 -I/usr/local/corex-4.4.0/include -include /usr/local/bin/corex_compat.h" \
  -DCMAKE_CUDA_FLAGS="-DSTRATA_CUDA_ARCH_NUM=600" \
  > /root/strata-cfg.log 2>&1
echo CONFIGURE_OK
cmake --build $BUILD --target strata strata-device strata-load -j 32 > /root/strata-build.log 2>&1
echo BUILD_OK
