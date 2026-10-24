# 预注册 08：减少重叠帧会让 SkyReels 回归吗？（2026-10-23，生成前冻结）

## 背景
- SkyReels-V2-DF 几乎不回归（预注册 06：20 秒时 α = 0.96），而且不受 CFG、历史帧噪声、去噪步数影响（实验记录 12）。
- 探索性检验（4 个场景）中，把重叠帧从 17 减到 5，20 秒时 α 从 0.89 降到 0.52。这支持"重叠帧构成的硬约束保住了开头"这一推测。
- 本实验在 12 个**未用于探索性检验**的场景上做确认。

## 设计
- **场景：** rx02、rx06、rx08、rx10、gx02、gx06、gx08、gx10、rx01、rx03、gx01、gx03。
- **条件：** 开头 L\* −16 和 +16；`overlap_history` 为 5 或 17；时长 20 秒；种子 0；通用提示词。
- **执行：** 每个场景的两种 overlap 在同一个任务里运行（arch_guard）。生成脚本 `sky_gen_x.py`，任务清单 `jobs/skyconf.jsonl`。共 48 个视频。
- **指标：** α = (p16 结尾 − m16 结尾) / (p16 开头 − m16 开头)，结尾取开头之后 [15, 20) 秒的 L\* 均值。

## 假设（`analyze_skyov.py`，冻结）
- **主要假设：** α(overlap 5) < α(overlap 17)。
- **检验方法：** 每个场景算出两种 overlap 的 α 之差，做双侧配对 t 检验，并用场景 bootstrap 给出 95% CI。判定标准：p < 0.05 且差 < 0。
- **次要分析：** 两种 overlap 在 5、10、15、20 秒的 α。

## 冻结文件（SHA-256）
```
75eaa574085cb55839e95c019c640b254d6e07f10d52dd1e9da88feddcf0b90a  analyze_skyov.py
4eb020bf429c1a6526d87e1b1496c3cb2a61dd1ffc36e08c583c31a3f54a7036  sky_gen_x.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
f3fe8b8a2e20278c297d9f1f1c27e332e4614c6b3278bd937525a742107c7276  jobs/skyconf.jsonl
```
