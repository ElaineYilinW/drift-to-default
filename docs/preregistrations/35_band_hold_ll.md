# 预注册 35：只固定 sink 的 j4 在 LongLive 上也去掉突跳吗？换成相邻的单个频率 j5 会怎样？（LongLive 与 Self Forcing，新种子，60 秒）（2026-10-29，生成前冻结）

## 背景
- 研究 34（实验记录 44）：在 Self Forcing 上，生成时只把 sink 的第 4 个时间频率（周期 33.5 个潜在帧）固定在距离 20，其他 21 个频率保持原始位置：
  - 突跳消失（+0.33 [+0.26, +0.39]）；
  - 60 秒的保留从 0.58 降到 0.16。
- 研究 34 留下两个问题：
  - **只在一个模型上：** LongLive 的突跳也跟着 sink 的位置走（研究 28），两个模型的探针形状几乎一样（r = 0.96）。如果突跳来自 j4，LongLive 上固定 j4 也应该去掉突跳。
  - **特异性：** 研究 34 的对照频带是 j8–11（4 个长周期频率），它让突跳略小了一点（−0.06）。"只有 j4"的说法需要和同样只固定一个频率的对照比较。频带探针上去掉相邻的 j5（周期 51.0）几乎不改变曲线（对比度 0.055 对全部频率的 0.058，相关 ≥ 0.99，主峰相同）。
- **数据状态：**
  - 冻结时本研究没有生成任何视频；种子 21 以前没有在这些场景上用过。
  - 冒烟测试只在场景 rx00、种子 99、15 秒、只有 m16 上运行（`jobs/s35_smoke.jsonl`）。它检查 LongLive-d 上"全部 22 个频率固定在 20"与 `dist:20` 逐帧一致、只固定 j4 从开头之后开始不同，没有看突跳或持续度。

## 设计
- **场景：** 预注册 06 的 24 个场景（rx00–rx11、gx00–gx11），通用提示词（`prompts/relax.jsonl`）。
- **模型：** sink 3，60 秒（243 个潜在帧，16 fps）。
  - LongLive：窗口 12。
  - Self Forcing DMD：窗口 21。
- **开头：** L\* 平移 −16 / +16（版本 m16、p16）。
- **种子：** 21。
- **生成器：**
  - LongLive：`ll_hold_gen_bandhold.py`（`ll_hold_gen_sinkpos.py` 的副本，只加了检查代码树），在 LongLive-d 上运行。LongLive-d 由 `jobs/setup_ll_d.sh` 建立，补丁是 `jobs/patch_ll_bandhold.py`，与 Self Forcing 的 `patch_sf_bandhold.py` 相同。
  - Self Forcing：`hold_gen_bandhold.py`，在 Self-Forcing-d 上运行（与研究 34 相同）。
- **条件（每个模型 3 个）：**
  - abs：原始位置的 sink；
  - j4：j4 固定在距离 20（`--sink_mode band:20:4`）；
  - j5：j5 固定在距离 20（`--sink_mode band:20:5`）。
  - 条件名：ll_abs、ll_j4、ll_j5、sf_abs、sf_j4、sf_j5。
- **执行：**
  - 每个场景 × 条件一个任务（两个版本），arch_guard 保证一对版本在同一种 GPU 上；
  - GPU 不限型号（≥ 40 GB，bf16）；
  - 只为 rx00、gx00 写视频文件；
  - 任务清单 `make_s35_tasks.py`（`jobs/s35.jsonl`，144 个任务），共 24 × 6 × 2 = 288 个视频。

## 指标
- **α[t0, t1)**：与研究 34 相同。
  - α = (end_p16 − end_m16) / (open_p16 − open_m16)；
  - open 是第 1–8 帧的 L\* 均值，end 是开头之后 [t0, t1) 秒的 L\* 均值；
  - 若 open_p16 − open_m16 < 4 L\* 则无定义；
  - 没有坍塌排除规则。
- **起点：** ORIG = 16.5、32.25、49.5 秒，与研究 25、28、34 相同。
- **突跳对比 C_arm：** 每个场景上，对各个起点 τ 取 α([τ + 0.5, τ + 2.5)) − α([τ − 2.5, τ − 0.5)) 的平均。

## 假设（`analyze_s35.py`，冻结；Holm 校正，共 4 个检验）
- **C1 LongLive 上固定 j4 去掉突跳：** C_ll_abs − C_ll_j4 > 0。
- **C2 LongLive 上没有突跳时保留得更少：** α(ll_abs, [50, 60)) − α(ll_j4, [50, 60)) > 0。
- **C3 Self Forcing 上不是任何一个频率都行：** C_sf_j5 − C_sf_j4 > 0。
- **C4 LongLive 上同样：** C_ll_j5 − C_ll_j4 > 0。
- **检验方法：**
  - 以场景为单位的双侧单样本 t 检验，Holm p < 0.05 且估计 > 0 为成立；
  - 95% CI 用场景 bootstrap（10,000 次）。
- **把握度：** 预期效应和研究 34 的 +0.33 同量级。
  - 研究 28 中 LongLive 原始位置的突跳对比是 +0.27 [+0.23, +0.31]，固定距离的 sink 在 −0.02 到 +0.02 之间。
  - 差值约 0.25、场景间标准差约 0.1 时，24 个场景的把握度接近 1。

## 次要分析
- **Self Forcing 上重复研究 34：** C_sf_abs − C_sf_j4；α(sf_abs) − α(sf_j4)（[50, 60) 秒）。
- **j5 与 abs 比较：** 每个模型的 C_j5 − C_abs，报告 90% t 区间和 ±0.10 的等价检验。
- **[50, 60) 秒的 α：** 每个条件与本模型的 abs 比较。
- **突跳细节：** 每个条件的 C 和每个起点的对比；10 秒窗口的 α 和 1 秒曲线；1 秒跳变超过 0.15 的时间。
- **与频带探针的相关：** 每个条件 1 秒 α 曲线的一阶差分，与频带探针在 sink 自然距离上、早 1 秒的份额的一阶差分的相关（t = 7–59 秒）。
  - 曲线对应："全部"对 abs，"去掉 j4"对 j4，"去掉 j5"对 j5。
  - 延迟 1 秒：研究 28 和研究 34（探索性）中都是延迟 1 秒最强。
- **motion：** [50, 60) 秒的 motion 和 motion < 1.5 的视频比例。

## 预测
- **C1 成立：** C_ll_abs 约 +0.25（研究 28：+0.27），C_ll_j4 约 0，差值约 +0.25。
- **C2 成立：** α(ll_abs, [50, 60)) 约 0.45（研究 28 的 ll_abs），α(ll_j4) 约 0.1–0.2，差值约 +0.25 到 +0.35。
- **C3 成立：** C_sf_j5 约 +0.25 到 +0.3（接近 abs），C_sf_j4 约 0，差值约 +0.3。
- **C4 成立：** 差值约 +0.25。
- **次要分析的预测：**
  - j5 与 abs 的差在 ±0.10 之内；
  - Self Forcing 上重复研究 34 的结果。
- **解读规则：**
  - **C1、C2、C3、C4 都成立：** 在两个模型上，突跳和 sink 多保留的部分来自 sink 的一个时间频率，相邻的频率没有这个作用。论文写"在两个模型中，只固定 sink 的一个旋转频率就去掉突跳和大部分保留，固定相邻的频率不会"。
  - **C1、C2 成立而 C3 或 C4 不成立：** j4 是必需的，但固定别的单个频率也会削弱突跳。论文只说 j4 必需，不说"只有 j4"。
  - **C1 不成立：** j4 的作用只在 Self Forcing 上得到证明，如实报告。
  - **C2：** 不论结果如何都报告估计值。

## 冻结文件（SHA-256）
```
c459eabab1849a97b031a8fa62bdc1d629e1460517e6b67e3b460c63e6bada20  analyze_s35.py
ed7afb93c56238d1c40a3f88444722cb0ab584ea75c7c0db48a77bade4ed2065  make_s35_tasks.py
f2d5cb4fdfe5bcd2fdae855cbc1ce6623e8763c8f16f7e8f13821fe8628c7523  jobs/s35.jsonl
9d722b0050f27b9ac382dda7e03266855822f9b8b6665b4c1075cdc164d60eb8  ll_hold_gen_bandhold.py
4eab5a269a3bdd683433e072c63aca147162ada5967cea2fc59e71b1ec5ae33f  jobs/patch_ll_bandhold.py
0f8f120879457b563e1779964d793bc4f8e166af4077c5a0fff14e4bae004007  jobs/setup_ll_d.sh
8bfac6050ca2c19c767851a254a00b0cb4233b4d4b3a02deed03a2abc960f965  LongLive-d/wan/modules/causal_model.py (patched)
6e9829a8b145d2dedcb81e8fdccb44404a7ed23de9608d1433633d1fa7d9076b  hold_gen_bandhold.py
dfd6acdf83bed9c0398afde6fe44f80595814eef5f1be8f9db5c0f67feacb707  jobs/patch_sf_bandhold.py
e6bb3c7f0059eec21d692a591e7fc594c986c70549743e43ef31f41c4861b858  Self-Forcing-d/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
fe8e39f6265453d29f81c619a5b760607fc7a22c1cc40a3c677dacda846cdcbb  results/corr/sinkband_out/sinkband.json
94efb91ff9bc4f4e3a7724338cb54fc3676dbd20ae65f9079dd273cff02608e0  jobs/s35_smoke.jsonl (smoke test only)
```
- 分析脚本已在合成数据上检查过：已知的突跳（abs、j5 +0.30 在两个起点，j4 无突跳）和持续度差值能正确恢复（C1 +0.20，C2 +0.35，C3、C4 恢复设定值）。
- 冒烟测试（LongLive-d，rx00、种子 99、15 秒、只有 m16）：`band:20:` 加全部 22 个频率与 `dist:20` 的逐帧 L\* 完全相同（最大差 0.0）；`band:20:4` 从第 9 帧（开头之后的第一块）开始与 `dist:20` 和原始位置都不同，说明改动只作用在开头之后。
