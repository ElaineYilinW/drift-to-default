# 预注册 11：非 Wan 模型 FramePack——开头的两条通道（2026-10-24，生成前冻结）

## 背景
- 到目前为止的所有模型（包括 SkyReels-V2-DF-1.3B）都来自 Wan2.1-1.3B。结论能否推广到其他底座，是最明显的疑问。
- FramePack（基于 HunyuanVideo，13B）专为长视频设计。它的每一段（33 帧）都通过**两条通道**以开头图像为条件：
  - **上下文通道：** 开头图像的 VAE 潜变量作为一个干净的前缀帧，放在 transformer 的上下文里；
  - **条件通道：** 开头图像的 SigLIP 嵌入作为全局条件（和文本提示词同一类）。
- 我们的理论是：条件（提示词）决定默认值 μ，上下文决定保留多少（α）。FramePack 让两条通道可以分开操纵：只在上下文里平移开头，或只在条件里平移开头。
- 预测：条件通道里的开头会移动默认值，所以只平移嵌入也会让视频变化（α_emb > 0）；两条都平移时保留最多。

## 设计
- **场景、提示词：** 预注册 08 的 12 个场景（rx02、rx06、rx08、rx10、gx02、gx06、gx08、gx10、rx01、rx03、gx01、gx03），通用提示词，种子 0，20 秒，30 fps，832×480。开头 L\* −16、+16。
- **条件：**
  - fpv_both：FramePack-F1（前向采样，和 KV-cache 模型一样按时间顺序生成），平移同时进入两条通道；
  - fpv_ctx：只平移上下文潜变量（SigLIP 嵌入用未平移的图像）；
  - fpv_emb：只平移 SigLIP 嵌入（上下文潜变量用未平移的图像）；
  - fpi_both：FramePackI2V（反向防漂移采样，从视频结尾往开头生成），两条通道都平移。
- **设置：** FramePack 演示的默认值：25 步，蒸馏引导 10，不用 CFG（`fp_gen.py`）。
- **执行：** fpv_both 与 fpv_ctx 在同一个任务里（arch_guard 保证同一架构）；fpv_emb、fpi_both 各自一个任务（`make_fp_tasks.py`，`jobs/fp.jsonl`）。共 12 × 4 × 2 = 96 个视频。
- **指标：** α(T) = (p16 结尾 − m16 结尾) / (p16 开头 − m16 开头)。开头取被平移的图像的 L\*（json 里的 open_L_shifted），结尾取开头之后 [T−5, T) 秒的 L\* 均值。

## 假设（`analyze_fp.py`，冻结；Holm 校正，共 3 个检验）
- **H1 FramePack 保留开头的大部分：** α(fpv_both, 20 秒) > 0.5。
- **H2 条件决定默认值：** α(fpv_emb, 20 秒) > 0。
- **H3 条件通道增加保留：** α(fpv_both, 20 秒) − α(fpv_ctx, 20 秒) > 0。
- **检验方法：** 以场景为单位的双侧单样本 t 检验，场景 bootstrap 95% CI。判定标准：Holm p < 0.05，且估计在预测的一侧。
- **次要分析：** 每个条件在 5、10、15、20 秒的 α；可加性 α(ctx) + α(emb) − α(both)（20 秒）；α(fpi_both) − α(fpv_both)（20 秒）。后两项跨任务，可能不在同一 GPU 架构上（只增加方差，不引入偏差）。
- **预测：** H1–H3 成立。

## 冻结文件（SHA-256）
```
31958d2820ace2a7ece0ace087aec4af66149588b7529c554e81cc4e775ba0f8  analyze_fp.py
e19888da3167a5190f28e1577474435b8d5bb5add5ea12c606a66194f2d03e3e  fp_gen.py
be7560b710eefea7726256d72d84e1f769f27b3f6497ceea6afdf94d687b90e2  make_fp_tasks.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
c3a391949b02d82fc0fa0cb44c6d066f3eb84052bf0acf00edb5a45e9e00a2a6  jobs/fp.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
