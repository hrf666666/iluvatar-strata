# Strata on Iluvatar CoreX(天数智芯 BI-V150)

[![GitHub repo](https://img.shields.io/badge/repo-hrf666666%2Filuvatar--strata-blue)](https://github.com/hrf666666/iluvatar-strata)
[![Upstream](https://img.shields.io/badge/upstream-Niko1221%2FStrata-green)](https://github.com/Niko1221/Strata)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-ggml%2Fllama.cpp-orange)](https://github.com/ggml-org/llama.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

把 **Strata**(为 **Qwen3.8-Flash-Next** 125B MoE 打造的开源推理引擎)移植到 **天数智芯 Iluvatar CoreX BI-V150** GPU,并附带同工具链的 **llama.cpp** 移植——在同样的硬件上忠实推理 **Qwen3.8-27B 全系量化版本** 与 **Flash-Next**。

- 🏠 本仓库:<https://github.com/hrf666666/iluvatar-strata>
- 🧭 上游 Strata:<https://github.com/Niko1221/Strata>(原版 README 存档于 [README-upstream.md](README-upstream.md))
- 📘 移植全记录(SASS 级取证、每个 bug 的定位与修复):[docs/ILUVATAR-PORT.md](docs/ILUVATAR-PORT.md)
- ✅ 实测输出证据:[docs/evidence/faithful-outputs.log](docs/evidence/faithful-outputs.log)

## 实测结果(单卡/双卡 BI-V150,2026-10-11)

| 模型 | 版本/量化 | 卡数 | 实测输出 | 速度 |
|---|---|---|---|---|
| Qwen3.8-27B | UD-Q4_K_M 16.5G | 1 | *"…[thinking] Answer **Paris**. Paris"* | 9.2 tok/s 生成 / 38 tok/s prefill |
| Qwen3.8-27B | UD-Q5_K_M 19.8G | 1 | *"Beijing is widely [recognized as the capital…]"*(思维链正确) | 11.3 tok/s 生成 / 42 tok/s prefill |
| Qwen3.8-27B | Q8_0 29G | 1 | *"…[thinking] **Paris**"* | 11.3 tok/s 生成 / 53.5 tok/s prefill |
| Qwen3.8-27B | **BF16 原版精度** 51.4G | 2 | *"…[thinking] **Paris**"* | 10.3 tok/s 生成 / 40.3 tok/s prefill |
| Qwen3.8-Flash-Next 125B | GSQ-RCO IQ2_XS 68G | 1(+CPU 专家) | 俳句创作逐词数音节 | 9.0 tok/s 生成 / 15.8 tok/s prefill |

> 支持的 27B 量化版本(unsloth/Qwen3.8-27B-GGUF 全家桶):`Q4_0 / Q4_1 / Q8_0 / UD-IQ1_M / UD-IQ1_S / UD-IQ2_XXS / UD-IQ2_S / UD-IQ3_XXS / UD-IQ3_S / UD-Q2_K_XL / UD-Q3_K_XL / UD-Q4_K_S / UD-Q4_K_M / UD-Q4_K_XL / UD-Q5_K_S / UD-Q5_K_M / UD-Q5_K_XL / UD-Q6_K / UD-Q6_K_L / UD-Q6_K_M / UD-Q6_K_XL / BF16`——
> ≤25G 单卡放得下;Q8_0/BF16 按显存用 `-ngl` 部分卸载或双卡 `--tensor-split`。所有量化走同一条已验证的 GGML CUDA 路径(见移植记录)。

## 快速开始

```bash
# 1. 工具链(需要 CoreX 4.4.0 + CoreX runtime,详见 docs/ILUVATAR-PORT.md)
sudo cp iluvatar-port/ivcc iluvatar-port/corex_compat.h /usr/local/bin/

# 2a. 构建 llama.cpp(27B 全系 + Flash-Next 忠实推理路径)
cd /path/to/llama.cpp && git apply iluvatar-port/llama.cpp-iluvatar.patch
iluvatar-port/build-llamacpp-iluvatar.sh          # -> build-iluvatar/bin/

# 2b. 构建 Strata 引擎(Flash-Next 原生引擎)
./strata-iluvatar.sh                              # -> build-iluvatar/strata

# 3. 跑 27B(健康卡;GPU 0/1 如被 XID 22 打挂需重启节点,务必 CUDA_VISIBLE_DEVICES 选卡)
CUDA_VISIBLE_DEVICES=2 ./build-iluvatar/bin/llama-cli \
  -m Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 99 -t 16 -n 64 \
  --temp 0 --top-k 1 -st -p 'The capital of France is'

# 4. 跑 Flash-Next 125B(GPU 上非专家 + CPU 专家池)
CUDA_VISIBLE_DEVICES=2 ./build-iluvatar/bin/llama-cli \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --override-kv 'tokenizer.ggml.pre=str:qwen35' \
  -ngl 99 -ot 'ffn_.*_exps\.weight=CPU' -t 32 -n 48 \
  --temp 0 --top-k 1 -st -p 'Write a haiku about rain.'

# 5. OpenAI 兼容 API
CUDA_VISIBLE_DEVICES=2 ./build-iluvatar/bin/llama-server \
  -m Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 99 --port 8902
```

Strata 引擎跑 Flash-Next(打包 + 生成)的完整流程见下文 [Strata 引擎](#strata-引擎flash-next-原生路径) 一节。

## 这场移植攻克了什么(简版,全记录在 [docs/ILUVATAR-PORT.md](docs/ILUVATAR-PORT.md))

1. **设备函数调用会打死 GPU**:CoreX 头文件的 fp16 helper 只是普通 `__inline__`,`-O0` 下生成真实设备调用,bi 后端把它降级为 `RPCMOV/PCSWAP`,硬件抛 **XID 22 非法指令**,该卡通道永久挂死。修复 = 全局 `-O1` 让一切内联(550 万行 SASS 零调用序列)。
2. **`double` 是假的**:后端把 FP64 静默降为 f32(`0.1+0.2` 得 f32 结果,`sqrt(-1)` 得 6.8e38 而非 NaN)——曾让所有归一化输出 NaN。设备端 double 全部改 float;`__dadd_rn` 系内建配 float 参数会出 Inf,一律换 `__f*`。
3. **64 线程硬件 warp 上的 3 参 shuffle**:无 width 的 `__shfl_sync(m, v, 0)` 广播读的是 64 线程组的 lane 0,**相邻 32 线程组两两塌缩**——Strata 超连接和 QSA 索引器正是这样坏的。全部 3 参 shuffle 改走 width=32 形式(逐位精确)。
4. **grid×block 的 z/y 乘积上限 ≈32768**:超限启动直接 invalid argument——binbcast 等超限时改走线性化 kernel。
5. **FA tile kernel 在 bi 上算错**(所有 batch>1 输出垃圾):dispatch 永不选 tile,vec kernel 补上缺失的越界 KV 掩码。FLASH_ATTN_EXT 校验 3555 失败 → 119(余项全是 bf16-KV;F16/F32/量化路径全绿,上表模型用的正是 F16 KV)。
6. **CUDA graph exec 更新**在 CoreX 10.2 返回未定义错误——降级为销毁重建。
7. dp4a/mma/cp.async/`%%globaltimer`/`mov.f32` 等指令被 bi 汇编器或寄存器分配器拒绝——dp4a 标量仿真、`<800` 老架构路径、profiler 时间戳置零等(MMQ 因编译器病态耗时整体编译出局,prompt 走反量化+cuBLAS,decode 保留 MMVQ)。
8. **CUDA 11/12/13 API → 10.2** 映射层 + 链接修复(`libcuda.so.1` 的 `Thunk_*` 符号在 `libixthunk`)。

## Strata 引擎(Flash-Next 原生路径)

```bash
export STRATA_GGUF_PY=/path/to/llama.cpp/gguf-py
python3 tools/iq_pack.py --gguf ...-00001-of-00002.gguf --out <packdir>
CUDA_VISIBLE_DEVICES=2 ./build-iluvatar/strata \
  --pack <packdir> --native ...-00001-of-00002.gguf \
  --ple-gguf ...-00002-of-00002.gguf \
  --spec 2 --prefill 512 \
  --expert-profile data/expert-profile.bin --expert-cache 2048 \
  --tokens '<ids>' --max-new 64
```

引擎加载 33G 专家池 + 3.2 亿行 PLE 表,MTP 投机验证全链路在 GPU 上跑,10–17 tok/s;kernel parity 全绿(`iq/s_gemv/mmvq_il/verify/kv_q8/cvec/sampler/gdn/gr`),唯一未解项是 logits 塌缩为常量 token(缺陷在 kernel 层之上:草稿器/头/嵌入交互,移植记录有定位线索)。Flash-Next 的**忠实**推理由上方 llama.cpp 路径承担。

## 相关项目(Qwen3.8-27B 生态,按星数)

| 项目 | Stars | 为什么相关 |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | ~80k | 本仓库的 27B/Flash-Next 忠实推理路径就是它的 CoreX 移植 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | ~50k | 生产级服务框架(PagedAttention、OpenAI API) |
| [unsloth/Unsloth](https://github.com/unsloth/Unsloth) | ~30k+ | 本仓库实测的全系 27B 量化(UD-Q4_K_M/Q5/Q6/Q8/BF16…)出自 Unsloth |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | ~20k | 高吞吐服务(RadixAttention、agentic 负载) |
| [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | ~20k+ | Qwen 官方仓库(模型卡、长上下文/YaRN 说明) |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | ~20k | 本仓库的上游——Flash-Next 消费级推理引擎 |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | ~10k+ | MoE 的 CPU+GPU 混合推理(Flash-Next 类模型的另一条路) |

模型权重:[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ·
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) ·
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)(hf-mirror.com 可直连)。

## 目录结构

```
strata-iluvatar.sh        Strata 的 CoreX 补丁+配置+构建(幂等)
iluvatar-port/
  ivcc                    CoreX clang CUDA 编译包装(-O1、架构、链接)
  corex_compat.h          CUDA 11/12/13 → 10.2 API 与代码生成 shim(强制包含)
  build-llamacpp-iluvatar.sh   llama.cpp 构建(产出 llama-cli / llama-server)
  llama.cpp-iluvatar.patch     llama.cpp 移植全补丁
src/ include/ tools/ …    Strata 源码(移植补丁已入树)
docs/ILUVATAR-PORT.md     移植工程全记录
docs/evidence/            实测输出与性能证据
```

## 已知边界

- FLASH_ATTN_EXT 剩 119/2520 失败,全部为 bf16-KV 用例(4.7%);F16/F32/量化 KV 全绿。
- MMQ(prompt 量化矩阵乘)整体编译出局(bi llc 病态耗时),27B prompt 走反量化+cuBLAS ≈38 tok/s。
- Strata 引擎 token 质量问题如上;其余组件(CPU 专家池、PLE、MTP 验证、图捕获)全部工作。
- **GPU 卫生**:一次 XID 22 会把那张卡挂死到节点重启。始终 `CUDA_VISIBLE_DEVICES` 选健康卡,卡死时查 `dmesg | grep XID` 与 `ixsmi`。

## License

Strata 为 MIT(见 [LICENSE](LICENSE));`iluvatar-port/` 各文件按同一许可提供。
