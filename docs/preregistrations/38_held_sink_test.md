# 预注册 38：规则选出的固定距离，在新开头、新种子和 4 个模型上，是否比重新对齐的 sink 保留更多、不突跳、动得不少？（2026-11-02，生成前冻结）

## 背景
- 研究 37 用一条固定规则为每个模型选出 sink 的固定距离 D\*：Self Forcing 66，LongLive 448，CausVid 和 GAN 由新的校准决定。
- 以前的确认只有 Self Forcing、距离 66、原来的 24 个场景（预注册 31）。本研究换成从没在 sink 研究里用过的 16 个开头和新种子，并把同一做法用到 4 个模型上。对照是 Rolling Forcing、Deep Forcing 那种重新对齐的 sink（紧贴窗口之前）。
- **数据状态：** 冻结时本研究没有生成任何视频。

## 设计
- **开头：** `prompts/heldtest.jsonl` 的 16 个：真实 NAVSIM 画面 nr01、nr04、nr06、nr09、nr11、nr14、nr16、nr18（按亮度排序每隔约 2.5 个取一个，L\* 约 22–58），生成画面 ng00–ng07。它们在预注册 03–05 中用过，但从没用于 sink 研究。
- **提示词：** 通用提示词（与预注册 06 相同）。
- **开头平移：** L\* −16 / +16（m16、p16）。**种子：** 5。**时长：** 120 秒（483 个潜在帧）。
- **模型：** Self Forcing DMD、CausVid、Self Forcing GAN（`hold_gen_sinkpos.py`，Self-Forcing-b，窗口 21），LongLive（`ll_hold_gen_sinkpos.py`，LongLive-c，窗口 12）；sink 3。没有 D\* 的模型不参加。
- **条件（每个模型）：**
  - held：sink 固定在 D\*（`--sink_mode dist:D*`）；
  - rr：sink 固定在重新对齐的距离（窗口 − 1：20，LongLive 11）；
  - abs：sink 在原始位置。
- **执行：** 每个模型 × 开头一个任务，跑 3 个条件和两个版本（arch_guard 保证同一种 GPU）。只为 nr01、ng00 写视频。任务清单由 `make_s38_tasks.py` 从研究 37 的 `dstar.json` 生成：Self Forcing、LongLive 现在生成（`jobs/s38_sf_ll.jsonl`，32 个任务），CausVid、GAN 在研究 37 校准完成后生成（`jobs/s38_cv_gan.jsonl`）。

## 指标
- α、motion、突跳对比的定义与预注册 31 相同。主要窗口是 [60, 120) 秒。
- 突跳起点：每个模型自己的探针曲线上突出度 ≥ 0.04 的尖峰 P（`analyze_sinkprobe_long.py` 的规则），起点 (P − 5) / 4 秒，取 62.5–117.5 秒之间的。Self Forcing 用 `sinkprobe_long_out.json`，LongLive 用 `llprobe_out.json`，CausVid、GAN 用研究 37 的探针。

## 假设（`analyze_s38.py`，冻结；对所有参加的模型一起做 Holm 校正，最多 12 个检验）
- **H1_m：held 比 rr 保留更多：** α(held, [60, 120)) − α(rr, [60, 120)) > 0。
- **H2_m：held 不突跳：** held 在该模型突跳起点处的突跳对比在 ±0.05 之内（TOST）。该模型在 62.5–117.5 秒没有起点时，H2_m 不做。
- **H3_m：held 动得不比 rr 少（非劣效，界值 0.5 灰度级）：** motion(held) − motion(rr) + 0.5 > 0，[60, 120) 秒。
- **检验方法：** 以开头为单位；t 检验双侧，Holm p < 0.05 且估计 > 0 为成立；TOST 取两个单侧 p 中较大的，Holm p < 0.05 为成立；95% CI 用开头 bootstrap（10,000 次）。
- **把握度：** 预注册 31 中 Self Forcing 的差值 +0.23，场景间标准差约 0.18；16 个开头时 d ≈ 1.3，在 0.004 水平上把握度约 0.95。

## 次要分析
- 每个模型和条件：[0, 60)、[60, 120) 秒和 10 秒窗口的 α，1 秒曲线，motion，冻结比例，清晰度比值，每个条件的突跳对比；
- held − abs 的 α 和 motion；坍塌计数。

## 预测
- **Self Forcing：** H1 成立，差值约 +0.2（held 约 0.4，rr 约 0.2）；H2、H3 成立。
- **LongLive：** H1 成立，差值约 +0.3（448 约 0.35，距离 11 约 0.05）；H2、H3 成立。
- **CausVid：** H1 可能成立但差值小（约 +0.1）；突跳起点可能很弱。
- **GAN：** H1 成立，差值约 +0.2–0.3。
- **解读规则：**
  - 若 4 个模型的 H1 大多成立，而 H2、H3 也成立：探针加少量校准就能为每个模型找到一个比重新对齐保留更多、不突跳、不冻结的 sink 位置，不需要训练。论文把它作为方法报告。
  - 若只在 Self Forcing 上成立：规则不能推广，论文如实写成"模型特定"。
  - 若 H3 不成立：保留更多以运动减少为代价，报告这一交换。

## 冻结文件（SHA-256）
```
1008f82648f96079bf8b9d87024b2b481d5defff34149b41e49d4a6372d9ed82  analyze_s38.py
63eed4085473f8e02e14d19904504c6a9bd99ef16ca08f1ede529b541cd2f839  make_s38_tasks.py
5aded6f546a31a3d7c26ef49fbce8e9b2e0dfe78e4b65ee5dee1c4eb1d25580c  select37.py
cbc24d57e382b4cf1a7586024aa04d2eba0c133a095a853c9a01bd955c61df5d  prompts/heldtest.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
073f21f1476aea62a192de6aa71b00159c2af78a44bfa2304f025b7ed65821d7  ll_hold_gen_sinkpos.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
fee24ef4bccf0bd7bc1ab277fcf15ed74e22c13344627ad1b77abfe068729873  jobs/dstar_sfll.json
0cfe007d7c956b621aeeafbd122ce89bc15c1d36a6507bbc3196d290e357b7c3  jobs/s38_sf_ll.jsonl
```
- 分析脚本已在合成数据上检查过：设定 held 0.45、rr 0.22、abs 0.43 时，H1 恢复 +0.23，H2 恢复约 0，H3 恢复约 +0.5。
