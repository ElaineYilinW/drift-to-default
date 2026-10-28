# 预注册 28：LongLive 的 sink 突跳是否也跟着旋转距离走？固定在一个距离上时，保留多少、动多少是否由 LongLive 自己的探针份额决定？（LongLive，60 秒）（2026-10-28，生成前冻结）

## 背景
- Self Forcing 上的结果：
  - 预注册 25（实验记录 31）：sink 往后移 18 或 36 个潜在帧，突跳提前 4.5 或 9 秒；固定在峰值距离 203 保留 0.86，但视频冻结，谷值 187 只保留 0.01。
  - 预注册 27：把 sink 固定在 17 个距离上，检验保留量和 motion 是否随探针份额变化（冻结，正在运行）。
- LongLive 是 Self Forcing 的长视频微调版本（窗口 12，sink 3，sink 在原始位置）。
  - 它的突跳与 Self Forcing 同时出现（预注册 10 的图谱）。
  - **设计检查（使用预注册 10 已有的数据）：** 用下文的突跳窗口，在场景平均的 1 秒曲线上，LongLive sink 3 的突跳对比约为 +0.24，Self Forcing 为 +0.34。
- **LongLive 的探针**（探索性，实验记录 30，`llprobe_out.json`）：
  - 设置与 Self Forcing 的第二次探针相同（开头 rx02、rx03、gx02、gx03，距离 3–723）。
  - Self Forcing 的 17 个主峰全部出现，曲线相关 r = 0.964。
  - LongLive 的份额整体高约 0.02（中位数 0.095 对 0.074）。
- 本研究在 LongLive 上做预注册 25 和 27 的干预，检验位置解释是否适用于第二个模型。

## 设计
- **场景：** 预注册 06 的 24 个场景（rx00–rx11、gx00–gx11），通用提示词（`prompts/relax.jsonl`）。
- **模型：** LongLive（基础权重 + LoRA），sink 3，窗口 12，60 秒（243 个潜在帧）。
  - 生成器 `ll_hold_gen_sinkpos.py`：`ll_hold_gen_long.py` 加上 `--probe`、`--sink_mode`，并在 npz 中记录 motion 和 open_diff（与 `hold_gen_sinkpos.py` 相同的定义）。
  - 在 LongLive-c 中运行：LongLive 1.0 的符号链接树，只有 `wan/modules/causal_model.py` 是实际文件，依次用 `jobs/patch_ll_probe.py` 和 `jobs/patch_ll_sinkpos.py` 打补丁。
  - 补丁只旋转 sink 键的一份拷贝（缓存保留原始的键），开头自己的上下文计算不变，与 Self Forcing 的 `patch_sf_sinkpos.py` 相同。
- **一致性检查**（冻结前完成，见下文）：同一块 GPU 上，原始 LongLive 树与 LongLive-c 不加开关、加 `off:0` 的输出逐位相同；加 `dist:30` 时不同。
- **开头：** L\* 平移 −16 / +16（版本 m16、p16）。
- **种子：** 0。
- **条件（20 个）：**
  - ll_abs：原始位置（不加开关）；
  - ll_off18：sink 往后移 18 个潜在帧（`--sink_mode off:18`）；
  - ll_off36：往后移 36 个潜在帧（`off:36`）；
  - ll_dDDD：sink 固定在距离 DDD（`dist:DDD`），17 个距离与预注册 27 相同：20、66、90、132、135、147、187、203、335、345、448、466、468、513、589、643、650。
- **LongLive 探针在这些距离上的份额：**

| D | 份额 | D | 份额 | D | 份额 |
|---|---|---|---|---|---|
| 20 | 0.124 | 147 | 0.105 | 466 | 0.134 |
| 66 | 0.121 | 187 | 0.077 | 468 | 0.144 |
| 90 | 0.131 | 203 | 0.182 | 513 | 0.104 |
| 132 | 0.130 | 335 | 0.167 | 589 | 0.063 |
| 135 | 0.167 | 345 | 0.080 | 643 | 0.150 |
| | | 448 | 0.088 | 650 | 0.068 |

- **执行：**
  - 每个场景一个任务，包含 20 个条件和两个版本（40 个视频）；arch_guard 保证一个场景的所有输出来自同一 GPU 架构；
  - 某个条件的 p16 输出已存在时跳过该条件，被中断的任务可以续跑；
  - GPU 不限型号（≥ 40 GB，支持 bf16）；
  - 只为 rx00、gx00 写视频文件；
  - 任务清单由 `make_s28_tasks.py` 生成（`jobs/s28.jsonl`，24 个任务）。
  - 共 24 × 20 × 2 = 960 个视频。

## 指标
- α[t0, t1) = (end_p16 − end_m16) / (open_p16 − open_m16)：
  - open 是第 1–8 帧的 L\* 均值；
  - end 是开头之后 [t0, t1) 秒的 L\* 均值；
  - 若 open_p16 − open_m16 < 4 L\*，则无定义。
- 没有坍塌排除规则。
- **motion[t0, t1)：** 逐帧 motion（与前一帧的平均灰度绝对差，0–100）在 [t0, t1) 秒上的均值，两个版本取平均。
- **share(D)：** LongLive 探针在距离 D 上的平均份额。
- **突跳起点**（与预注册 25 相同）：
  - τ_k(δ) = (P_k − 5 − δ)/4 秒，P = 71、134、203、266（两个模型的探针都有这 4 个峰）；
  - 突跳窗口 W_k = [τ_k + 0.5, τ_k + 2.5) 秒，只保留 [5, 60) 秒内的窗口；
  - A_X(δ)：条件 X 在 δ 对应的所有窗口上的 α 均值（每个场景一个值）。

## 假设（`analyze_s28.py`，冻结；Holm 校正，共 4 个检验）
- **L1 突跳跟着 18 帧的移动走：** [A_off18(18) − A_off18(0)] − [A_abs(18) − A_abs(0)] > 0。
- **L2 突跳跟着 36 帧的移动走：** [A_off36(36) − A_off36(0)] − [A_abs(36) − A_abs(0)] > 0。
- **L3 保留量随份额上升：** 每个场景跨 17 个固定距离的 Spearman ρ_s(share, α[50, 60))（至少 12 个有定义的距离），场景平均 > 0。
- **L4 motion 随份额下降：** 每个场景的 ρ_s(share, motion[50, 60))，场景平均 < 0。
- **检验方法：**
  - 以场景为单位的双侧单样本 t 检验；
  - Holm p < 0.05 且估计的符号与假设一致为成立；
  - 95% CI 用场景 bootstrap（10,000 次）。

## 次要分析
- [40, 50) 秒窗口的 L3、L4；
- 每个条件在自己起点上的突跳对比（ll_abs、ll_off18、ll_off36 用各自的 δ；固定距离的条件用 δ = 0）；
- 1 秒曲线上大于 0.15 的跳变时刻；
- 每个条件的：
  - α（10 秒窗口和 1 秒曲线）；
  - [50, 60) 秒的 motion、open_diff；
  - motion < 1.5 的视频比例；
- 跨距离：场景平均 α、motion 与 LongLive 份额、Self Forcing 份额的 Pearson 和 Spearman 相关；
- 探针对照（与预注册 26 相同）：1 秒 α 曲线的一阶差分与 LongLive 探针份额在 t − lag 的一阶差分的相关，lag = 0–3 秒，t = 6–59 秒，ll_abs、ll_off18、ll_off36 用各自的 δ；
- 有预注册 27 的结果时：17 个距离上 LongLive 与 Self Forcing 的平均 α、motion 的相关。

## 预测
- **L1、L2 成立。** LongLive 的突跳比 Self Forcing 弱（设计检查 +0.24 对 +0.34），预计 L1 约 +0.3、L2 约 +0.15。
- **L3、L4 成立：** 平均 ρ 分别约 +0.5 到 +0.8 和 −0.3 到 −0.7。
  - 把握比 Self Forcing 低：LongLive 的份额整体更高，但它原始位置的 sink 在 60 秒时保留的反而比 Self Forcing 少（0.45 对 0.61），所以 LongLive 的保留量可能对份额不那么敏感。
- **解读规则：**
  - 若 L1–L4 都成立：位置解释（突跳由 sink 的旋转距离决定，保留量由该距离上的注意力份额决定）在第二个模型上成立，包括它的微调没有改变的探针峰。
  - 若 L1、L2 成立而 L3 不成立：突跳的时刻在 LongLive 上同样由距离决定，但固定距离时的保留量不由份额决定（例如 LongLive 的训练让它对 sink 的依赖与位置无关）。
  - 若 L1、L2 不成立：LongLive 的突跳虽然时刻相同，但不能用移动 sink 来移动；论文中的位置解释限于 Self Forcing。

## 一致性检查（冻结前，2026-10-28）
- `jobs/run_llsp_check.sh`：同一块 GPU（A40）上 rx00、L\* −16、sink 3、种子 0、10 秒，比较逐帧 L\* 均值：
  - LongLive-c 不加开关：与原始 LongLive 树逐位相同（最大差 0.0）；
  - 加 `off:0`（不旋转）：逐位相同；
  - 加 `dist:30`：不同（最大差 15.4 L\*），说明开关起作用。距离 30 不在本研究的距离中，这 10 秒只用于核对。

## 冻结文件（SHA-256）
```
0938e2f5d25a3c345179ffacb61021932e4db626093e7c07f8a76f9ca0ffec58  analyze_s28.py
62eb515368cf774a98124831ced96ad10cc617e5a4ca81f7f9601e9d605f5360  make_s28_tasks.py
aa9b610bbb50c2b20e2103353baa66528e536de9f8fef1736730cba8465483f9  jobs/s28.jsonl
ea6a93de994e02811cfdc2cd406db6856fd84b6bc255ff2369b008837402f9ce  s27_distances.py
073f21f1476aea62a192de6aa71b00159c2af78a44bfa2304f025b7ed65821d7  ll_hold_gen_sinkpos.py
683048690f76f5845dfbf32b87354e89c6d7d9a579797b20f9f9aa30bda2be7d  jobs/patch_ll_sinkpos.py
f4e2146c9b4443fe68b187fca2c2ce058cab3ab99e45ca36e449d8dfd4b2b2f9  jobs/patch_ll_probe.py
4745cce792fdc31a0442245b9ff6596d06288bac08bf67a807e142d702d660a5  LongLive-c/wan/modules/causal_model.py (patched)
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
a84554132486503056181f94cad78b3e04c86108d920b5c4c51cee92b3c790eb  jobs/run_llsp_check.sh
89d1007ae1d6165b5027d0d681bdf54cf6a9fb7decd32ecbcdd37ee8db2c81f9  llprobe_out.json (exploratory probe result, LongLive)
64385a87e26b2110501e89abee7f07196cc093372da13b57f11f5df26cdde923  sinkprobe_long_out.json (exploratory probe result, Self Forcing)
```
