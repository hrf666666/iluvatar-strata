<h1 align="center">Strata on Iluvatar</h1>

<p align="center"><b>在天数智芯 BI-V150 显卡上运行 Qwen3.8-Flash-Next 125B、Qwen3.8-27B 全系和 GLM-5.3</b><br>
纯国产 GPU · Linux · 免费开源</p>

<p align="center">
<a href="https://github.com/hrf666666/iluvatar-strata">仓库主页</a> ·
<a href="https://github.com/Niko1221/Strata">上游 Strata</a> ·
<a href="docs/ILUVATAR-PORT.md">移植全记录</a> ·
<a href="docs/evidence/faithful-outputs.log">实测输出</a>
</p>

这是 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp) 在 **天数智芯 Iluvatar CoreX BI-V150** 上的完整移植。不需要 NVIDIA 或 AMD 的卡:Qwen3.8-Flash-Next 125B 在一张 32 GB 的 BI-V150 上聊天、写代码、数俳句音节;Qwen3.8-27B 全系量化和 GLM-5.3-Flash / GLM-5.3 开箱即用,输出忠实(每个模型的实测文本见 [docs/evidence/](docs/evidence/))。

## 有多快?

一个 token 大约是 ¾ 个词。**写答案**是回复出现的速度,**读提示**是你发来的长文被吃进去的速度。测试机:单张 BI-V150(32 GB),2026-10-12。

| 模型 | 大小 | 写答案 | 读提示 |
| --- | --- | ---: | ---: |
| **Qwen3.8-27B** | UD-Q4_K_M(16.5 GB) | 11.4 tokens/s | 232.5 tokens/s |
| Qwen3.8-27B | BF16 原版精度(51.4 GB,双卡) | 10.3 tokens/s | 40.3 tokens/s |
| **Qwen3.8-Flash-Next 125B** | GSQ-RCO IQ2_XS(68 GB) | 9.0 tokens/s | 15.8 tokens/s |
| GLM-5.3-Flash 320B | Q4_K(181 GB) | 7.5–8.2 tokens/s | 10–11 tokens/s |
| **GLM-5.3 大杯** | UD-IQ1_M(228 GB) | 4.8–5.0 tokens/s | 8.6–9.8 tokens/s |

125B 和 320B 的模型比显存大:显卡装下常用部分,其余专家放内存和 CPU 里跑——这正是 Strata 的设计。

**要吞吐而不是单流速度?** 27B 一张卡开 64 路并发是 46.3 tokens/s,四张卡各跑一个实例实测 **167.4 tokens/s**(单流的 18.2 倍):

```bash
./iluvatar-port/serve-parallel.sh Qwen3.8-27B-UD-Q4_K_M.gguf "3,4,6,7" 8960 64 512
```

测量过程和证据:[docs/evidence/perf-optimization.log](docs/evidence/perf-optimization.log)。

## 你需要什么

| | |
| --- | --- |
| **显卡** | 天数智芯 BI-V150(32 GB)一张起。大模型可以多卡或 CPU 分担 |
| **内存** | 27B 要 32 GB;Flash-Next 125B 要 64 GB;GLM-5.3 320B 要 1 TB |
| **硬盘** | 模型大小 + 20 GB(27B 约 20 GB,Flash-Next 约 70 GB,GLM-5.3 大杯约 230 GB) |
| **系统** | Linux,CoreX 4.4.0 工具链 + CoreX runtime(驱动 4.4.0 / CUDA 10.2) |
| **27B 量化** | unsloth 全家桶都支持:Q4_0 … Q8_0、UD-IQ1_M … UD-Q6_K_XL、BF16;≤25 GB 单卡放得下 |

## 安装

```bash
git clone https://github.com/hrf666666/iluvatar-strata.git && cd iluvatar-strata

# 1. 装编译包装(把 ivcc 和兼容头放进系统)
sudo cp iluvatar-port/ivcc iluvatar-port/corex_compat.h /usr/local/bin/

# 2. 构建 llama.cpp(27B / Flash-Next / GLM 的推理引擎)
git clone https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
cd /path/to/llama.cpp && git apply <你的路径>/iluvatar-port/llama.cpp-iluvatar.patch
<你的路径>/iluvatar-port/build-llamacpp-iluvatar.sh     # -> build-iluvatar/bin/

# 3. 构建 Strata 引擎(Flash-Next 原生引擎,可选)
cd <你的路径> && ./strata-iluvatar.sh                   # -> build-iluvatar/strata
```

`ivcc` 是 CoreX clang 的编译包装:强制 `-O1`(见下方"它是怎么跑起来的")、指定架构、修好链接。补丁只改 `ggml-cuda` 一个目录,20 个文件,对干净树验证过可以 `git apply`。

## 使用

```bash
# 27B:忠实推理,一句话的事
CUDA_VISIBLE_DEVICES=3 ./build-iluvatar/bin/llama-cli \
  -m Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 99 -t 16 -n 64 \
  --temp 0 --top-k 1 -st -p 'The capital of France is'

# Flash-Next 125B:显卡跑非专家,CPU 专家池跑其余
CUDA_VISIBLE_DEVICES=3 ./build-iluvatar/bin/llama-cli \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --override-kv 'tokenizer.ggml.pre=str:qwen35' \
  -ngl 99 -ot 'ffn_.*_exps\.weight=CPU' -t 32 -n 48 \
  --temp 0 --top-k 1 -st -p 'Write a haiku about rain.'

# GLM-5.3-Flash 320B(--cpu-moe 同理;GLM-5.3 大杯用 UD-IQ1_M 六分片)
CUDA_VISIBLE_DEVICES=3 ./build-iluvatar/bin/llama-cli \
  -m GLM-5.3-Flash-Q4_K-00001-of-00002.gguf --cpu-moe -ngl 99 -c 4096 -t 32 \
  --temp 0 --top-k 1 -st -p 'The capital of France is'

# OpenAI 兼容 API(单卡;高吞吐见"有多快"里的 serve-parallel.sh)
CUDA_VISIBLE_DEVICES=3 ./build-iluvatar/bin/llama-server \
  -m Qwen3.8-27B-UD-Q4_K_M.gguf -ngl 99 --port 8902
```

Strata 原生引擎(打包 + 生成)的完整用法见 [docs/ILUVATAR-PORT.md](docs/ILUVATAR-PORT.md#11-breakthrough-the-strata-engine-emits-faithful-text-2026-10-12)。

## 出问题了?

- **某张卡一用就挂死。** 这张卡大概率被 XID 22 打挂过,重启节点前救不回来。用 `CUDA_VISIBLE_DEVICES` 只选健康卡(`ixsmi` 看状态,`dmesg | grep XID` 查原因)。
- **编译时编译器吃满 CPU 很久不结束。** 确认走的是 `ivcc`(它强制 `-O1`);`-O0` 会生成设备函数调用,那正是 XID 22 的来源。
- **多卡没变快。** 层切分(`--tensor-split`)对单流 decode 无增益,这是带宽物理决定的。要吞吐用 `serve-parallel.sh` 每卡一个实例。
- **输出是乱码或 NaN。** 检查 KV 类型:F16 / F32 / 量化 KV 全绿;bf16-KV 有 4.7% 的用例仍失败(已知边界)。Strata 引擎需要 `STRATA_PREFILL_MMQ=0 STRATA_VERIFY_T1=1`。

## 它是怎么跑起来的?

CoreX 的 CUDA 是 10.2,后端叫 bi,和主流 NVIDIA 卡不完全一样。这次移植踩过的坑,挑最大的四个:

- **设备函数调用会打死 GPU。** `-O0` 下 fp16 帮手函数成了真实调用,bi 把它降级成 `RPCMOV/PCSWAP`,硬件直接抛 XID 22。`ivcc` 强制 `-O1` 让一切内联,550 万行 SASS 零调用序列。
- **`double` 是假的。** 后端把 FP64 静默当 f32 算,归一化全出 NaN。设备端 double 全部改 float。
- **64 线程硬件 warp。** 三参数 shuffle 广播会跨 32 线程组塌缩,全部改走 width=32 形式,逐位精确。
- **dp4a:汇编器拒绝 PTX,但硬件有指令。** 内联 `asm("dp4a…")` 报"invalid operand",`__dp4a()` 内建却是真的、bit-exact 的。恢复它之后 decode 提速 24%、prefill 提速 5.9 倍(MMQ 量化矩阵乘随之回归)。

完整的 SASS 级取证、每个 bug 的定位与修复、性能优化的全部测量:[docs/ILUVATAR-PORT.md](docs/ILUVATAR-PORT.md)。输出证据(每个模型的真实生成文本):[docs/evidence/](docs/evidence/)。

## 致谢与许可

模型来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 团队、智谱 GLM 团队,量化出自 [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) 和 [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)。本仓库基于 [Strata](https://github.com/Niko1221/Strata)(上游 README 存档:[README-upstream.md](README-upstream.md))和 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)。MIT 许可,见 [LICENSE](LICENSE)。
