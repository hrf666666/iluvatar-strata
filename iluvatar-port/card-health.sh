#!/usr/bin/env bash
# 卡健康探针：对每张卡跑平凡内核，40s 超时即视为坏卡（上下文创建挂死）。
# 用法: bash card-health.sh [卡号...]   (缺省扫 0..7)
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/iluvatar/lib64:${LD_LIBRARY_PATH:-}
probe=/private/iluvatar-strata/bin/ivcard-probe
[ -x "$probe" ] || { echo "ivcard-probe missing"; exit 1; }
cards=${*:-0 1 2 3 4 5 6 7}
for c in $cards; do
  if out=$(CUDA_VISIBLE_DEVICES=$c timeout 40 "$probe" 2>&1 | tail -1); [ "$out" = "bad=0  PASS" ]; then
    echo "card $c: OK"
  else
    echo "card $c: DEAD (context creation hangs; ixsmi will NOT show it)"
  fi
done
