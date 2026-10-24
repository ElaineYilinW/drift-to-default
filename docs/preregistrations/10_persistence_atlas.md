# 预注册 10：60 秒持续度图谱，以及 sink 大小、位置编码的作用（2026-10-24，生成前冻结）

## 背景
- 预注册 06 在 31 秒比较了 14 个配置的持续度 α。但预注册 07 显示，31 秒时 Self Forcing 还没到平衡（有 sink：31 秒 0.28，60 秒 0.58；无 sink：0.23 → 0.03）。所以 31 秒的排序可能不是平衡时的排序。
- 两个异常还没解释：
  - sink 1 比没有 sink 保留得更少（31 秒，探索性）；
  - 有 sink 时，α 在 31.5 秒到 34.5 秒之间跳升（预注册 07 的曲线）。
- 一个候选解释：Self Forcing 的 sink 保留原来的绝对 RoPE 位置，距离当前帧越来越远，超出训练范围。把 sink 的位置编码重新旋转到窗口前面（Rolling Forcing、Deep Forcing 的做法）可以检验这一点。
- 本研究把所有模型放到 60 秒重新测 α，并检验 sink 大小和 sink 位置编码。

## 设计
- **场景、提示词：** 与预注册 06 相同的 24 个场景、通用提示词、种子 0；60 秒（243 个潜在帧）。
- **配置：**
  - Self Forcing DMD：sink 3、sink 6、sink 3 重新旋转（一个任务）；无 sink、sink 1、sink 1 重新旋转（另一个任务）。开头 L\* −16、+16。窗口都是 21。sink 6 = 开头的块 + 第一个生成的块。
  - LongLive：sink 3、无 sink；Self Forcing GAN、SiD、SiD v2、10 秒训练版；Rolling Forcing。开头 L\* −16、0、+16。
- **执行：** 互相比较的臂在同一个任务里，开始时运行 arch_guard，保证在同一 GPU 架构上（`make_atlas_tasks.py`，任务清单 `jobs/atlas.jsonl`）。生成脚本：`hold_gen_long_rr.py`（hold_gen_long.py 加 `--rerotate`）、`ll_hold_gen_long.py`、`rf_gen_long.py`（rf_gen.py 加分段解码）。共 24 × (12 + 6 + 6 + 6 + 3) = 792 个视频。
- **指标：** α(T)：在同一场景内，把开头之后 [T−10, T) 秒的 L\* 均值对开头的 L\*（第 1–8 帧）做 OLS，取斜率（2 个或 3 个版本）。主要终点 T = 60 秒。

## 假设（`analyze_atlas.py`，冻结；Holm 校正，共 4 个检验）
- **H1 更深的 sink 保留更多：** α(sf60_s6) − α(sf60_s3) > 0。
- **H2 任何 sink 都保留一些：** α(sf60_s1) − α(sf60_s0) > 0（链式递推模型的预测；31 秒的探索性结果方向相反）。
- **H3 LongLive 的 sink 在 60 秒仍然保留：** α(ll60_s3) − α(ll60_s0) > 0。
- **H4 sink 的位置编码有影响：** α(sf60_s3rr) − α(sf60_s3) ≠ 0（双侧，不预测方向）。
- **检验方法：** 每个场景的差值，双侧单样本 t 检验，场景 bootstrap 95% CI。H1–H3：Holm p < 0.05 且估计 > 0；H4：Holm p < 0.05（报告方向）。
- **次要分析：**
  - α(sf60_s1rr) − α(sf60_s1)；
  - 每个配置在 10–60 秒的 α；60 秒 α 与预注册 06 的 31 秒 α 的 Spearman 相关；
  - 每个 Self Forcing DMD 臂的跳升：α(34.5 秒) − α(31.5 秒)（1 秒窗口）；
  - 开头在前 4.5 秒内丢失的比例 1 − α(4.5 秒)，与 60 秒总丢失比例对比（区分"一开始就拒绝人为平移"和"慢慢遗忘"）；
  - 三版本配置在 60 秒的零漂移水平。
- **预测：** H1–H3 成立；H4 没有方向预测。

## 冻结文件（SHA-256）
```
6891bdb856b271966065ad5ba6ba86beaf6dfdc8d4348086ada66d8f8b25db28  analyze_atlas.py
9cdf3d348ca05116fd74e0c9cd5b87ccc5a26529b1700a97257570f6c09b7247  make_atlas_tasks.py
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
8fab4d7b9f578e65083fb0a5a4c1fac3893a2b13ae6712108e1367821c7f36a8  jobs/atlas.jsonl
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
```
