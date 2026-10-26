# 预注册 18：MAGI-1——非 Wan 架构的自回归模型，注意力窗口与完整历史（2026-10-25，生成前冻结）

## 背景
- 论文的主要局限是 7 个模型中有 6 个基于 Wan2.1。FramePack 是另一个基座，但它不是逐块 KV cache 的模型。
- MAGI-1（Sand AI，4.5B 蒸馏版）是自己架构的自回归视频模型：
  - 每块 6 个潜在帧（24 帧，24 fps 下 1 秒）；
  - 每块对之前的块做注意力（KV cache）。
- 注意力范围由运行配置决定：
  - **发布的设置（mgw）：** `noise2clean_kvrange [5, 4, 3, 2]`、`clean_chunk_kvrange 1`，即最多看之前 4 块（约 4 秒），没有 sink。开头几秒后就离开上下文。
  - **完整历史（mgf）：** `noise2clean_kvrange []`（代码的默认范围），每块看之前所有的块，开头始终在上下文中。
- 我们的理论：只靠 KV cache 窗口的模型会遗忘开头；开头一直留在上下文中时会保留一部分。
- **冒烟测试**（rx00，未平移，30 秒，A40，非本研究场景）：
  - mgw 约 12 秒后画面冻结（L\* 不再变化）；
  - mgf 持续运动，20 秒时内容已变成通用提示词的城市街道；
  - 耗时：mgw 1.5 小时，mgf 2.3 小时。
  - 冻结是发布模型在长时间生成下的行为，按预先规定的指标记录（次要分析），不作排除。

## 设计
- **场景：** 预注册 08 的 12 个场景，开头 L\* −16 / +16，通用提示词，种子 0，30 秒，分辨率 432×768。
- **生成：** `magi_gen.py`（MAGI-1 仓库 commit 0fcefde，未修改）。
  - flash-attn 用官方预编译包 2.7.4.post1；
  - flashinfer 只用到 `silu_and_mul`，由 `shims_magi/flashinfer` 用 PyTorch 实现，数学相同。
- **执行：**
  - 每个场景 × 条件一个任务（两个版本，arch_guard）；
  - 另有通用提示词的从零生成（文生视频），每个条件种子 0–3；
  - 所有任务限定在 A100 组（A100、A800）；
  - 任务清单 `jobs/magi.jsonl`（`make_magi_tasks.py`）。
  - 共 48 + 8 = 56 个视频。
- **指标：**
  - α(T) = (p16 结尾 − m16 结尾) / (p16 开头 − m16 开头)；
  - 开头取第 1–8 帧的 L\* 均值；
  - 结尾在 T ≥ 20 时取 [T−10, T) 秒的 L\* 均值，T < 20 时取 [T−1, T)。
- **坍塌规则：** 任一版本在 [0, 30) 秒内有任何 1 秒 L\* 均值低于 3 或高于 97，该场景从该条件中排除。

## 假设（`analyze_magi.py`，冻结；Holm 校正，共 3 个检验）
- **H1 发布的窗口设置会遗忘：** α(mgw, 30) − α(mgw, 5) < 0。
- **H2 完整历史比窗口保留更多：** α(mgf, 30) − α(mgw, 30) > 0。
- **H3 完整历史保留一部分：** α(mgf, 30) > 0。
- **检验方法：**
  - 以场景为单位的双侧单样本 t 检验，场景 bootstrap 95% CI；
  - Holm p < 0.05 且估计在预测的一侧为成立。

## 次要分析
- 每个条件在 1、2、5、10、20、30 秒的 α。
- 冻结：若视频在 [20, 30) 秒内平均 L\* 的逐帧变化绝对值均值低于 0.01，记为冻结。报告每个条件的冻结比例，并在排除冻结场景后重做 H1–H3。
- 每个条件的吸引子偏移：从零生成视频在 [20, 30) 秒与 [0, 2) 秒的 L\* 均值之差。
- 每个条件的零漂移水平（由两个版本求出），以及 |结尾 − 开头| 的均值。

## 预测
- **H1 成立。**
- **H2 成立。**
- **H3 不确定：** 冒烟测试中完整历史下内容仍然回到城市，色调保留多少未知。
- **冻结：** mgw 的冻结比例高于 mgf。

## 冻结文件（SHA-256）
```
cb7842463b06265d77a85de8c463943c9f8439f02fb270d0c2ea53f8a34f936d  analyze_magi.py
24f47bf25724ddb8166f08e8fe9c167e33a1a57a76cad185ade223ac01c6ba55  make_magi_tasks.py
bd5b465ccfd8a7492637702f20bb40ddd90f9f494a0fe00d5a8baa990d61313e  magi_gen.py
ff2a3e559b4bd5e5ab074a4ea30e2392b3bced2e85cc6db6fb106ced29418a5b  shims_magi/flashinfer/__init__.py
7e9625548bdc661f0c6f4b1a0cc8c45f833de5a9044da31091c1510fd72d8ef1  shims_magi/flashinfer/activation.py
23f471e6be9a8985e46c349c252b798e1b10228fe25f009962eb470c740a5242  shims_magi/flashinfer/gemm.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
a2ffe0864c446290dbfb5876a7407b9c32eceb0c050333ffa4893c184f72be8e  jobs/magi.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
- MAGI-1 仓库：https://github.com/SandAI-org/MAGI-1 commit 0fcefdef8ce2df37a3b8890979433c06eb003328；权重 sand-ai/MAGI-1 ckpt/magi/4.5B_distill、ckpt/t5、ckpt/vae。
