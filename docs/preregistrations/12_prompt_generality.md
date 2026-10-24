# 预注册 12：换成非驾驶场景的提示词，回归和 sink 效应还在吗？（2026-10-24，生成前冻结）

## 背景
- 预注册 06、07、10 的干预实验都只用了一个通用驾驶提示词。回归和 sink 效应是否只是驾驶场景的特性，是一个明显的疑问。
- 灯光实验（实验记录 06）的探索性再分析显示，Self Forcing 的持续度随提示词变化（晴天 0.34、阴天 0.24、黄昏 0.17、夜晚 0.05；交互项 cluster-robust p = 0.001）。本研究在新的场景类型上，预先登记检验"持续度随提示词变化"。

## 设计
- **提示词：** 4 个非驾驶场景：公园步行、海岸航拍、厨房、户外市场（`prompts/pgen.jsonl`），模板与通用驾驶提示词相同（"…are visible, and the camera moves … with natural lighting and realistic colors."）。
- **开头：** Self Forcing DMD 用每个提示词从零生成 5 秒视频（种子 0–5），取第 64 帧（4 秒）。不做筛选（`pgen_openings.py`）。共 24 个开头。
- **续写：** 每个开头在自己的提示词下续写，开头 L\* −16、0、+16，种子 0。
  - Self Forcing DMD：sink 3、无 sink，60 秒（`hold_gen_long.py`），同一个任务；
  - LongLive：sink 3、无 sink，31 秒（`ll_hold_gen.py`），同一个任务。
  - 每个任务开始时运行 arch_guard（`make_pgen_tasks.py`）。共 24 × 3 × 4 = 288 个视频。
- **指标：** α(T)：在同一开头内，把开头之后 [T−10, T) 秒的 L\* 均值对开头的 L\*（第 1–8 帧）做 OLS，取斜率。Self Forcing T = 60 秒，LongLive T = 30 秒。

## 假设（`analyze_pgen.py`，冻结；Holm 校正，共 3 个检验）
- **H1：** α(sf60_s3) − α(sf60_s0) > 0（60 秒）。
- **H2：** α(ll_s3) − α(ll_s0) > 0（30 秒）。
- **H3 持续度随提示词变化：** 4 个场景之间，每个开头的 α(sf60_s3, 60 秒) 的单因素方差分析（不预测方向）。
- **检验方法：** H1、H2 以开头为单位做双侧单样本 t 检验，Holm p < 0.05 且估计 > 0 为成立；H3 Holm p < 0.05 为成立。95% CI 在场景内重抽开头。
- **次要分析：** 每个场景、每个配置在 10–60 秒的 α；每个场景的零漂移水平；LongLive（30 秒）上重复 H3。
- **预测：** H1–H3 成立。

## 冻结文件（SHA-256）
```
0254d401b63def9881cdfe8309d854d0dcc815cb7d7b02faf4caefd6bc9e07c7  analyze_pgen.py
9c4025d80bcb7c76b9a8b9278ea1f6ac8fb3319472f87c202002841d1f58f987  pgen_openings.py
41cd031fea5d5e59103eab1979a0dd3a7cfc45e2818762edf7c13bc4fcd91edf  make_pgen_tasks.py
4f14ab0f30365c1f75c004e7b676712d3516010d864531d0663e248b140ce1a4  hold_gen.py
d468456274cfc794150a3f8252b28b5c261568f95d0451c14307bcaa44a11d64  hold_gen_long.py
698c02be1ab51f4e788561a98011c43a847ab54c03ec8caea658ff61c5c2e36b  ll_hold_gen.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
9e3730dae70832f9be51c5c5e75e2b193c87626768038b8117f4b5b6701ba3f8  prompts/pgen.jsonl
60779f2a1ddad8426fd053aae4a062261e8eadc947d86ab7ed6a496bb819cf34  jobs/pgen_a.jsonl
```
- 第二阶段任务清单 `jobs/pgen_b.jsonl` 由 `make_pgen_tasks.py b` 在开头生成后得到，内容完全由冻结脚本和开头决定。
