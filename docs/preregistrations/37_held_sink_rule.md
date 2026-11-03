# 预注册 37：用一条固定规则为每个模型选出 sink 的固定距离（Self Forcing、LongLive 用已有数据；CausVid、Self Forcing GAN 新做校准）（2026-11-02，生成前冻结）

## 背景
- 预注册 27、31：Self Forcing DMD 的 sink 固定在距离 66 时，第 3 分钟保留 0.45，重新旋转只有 0.22，且没有突跳、不冻结。
- 预注册 28：同一个距离 66 在 LongLive 上只保留 0.17；LongLive 在距离 448 保留 0.36、冻结 4%。所以距离不能直接迁移，要按模型选。
- 本研究把"选距离"写成一条事先固定的规则，用在 4 个模型上。研究 38 在新场景、新种子上检验规则选出的距离，研究 39 在不给开头的 60 秒视频上用 VBench 式指标评测。
- **数据状态：** 冻结时 CausVid 和 GAN 的校准视频都没有生成。两个模型的注意力探针（探索性，与 `jobs/run_sinkprobe_long.sh` 相同的设置：rx02、rx03、gx02、gx03，6 段，距离 3–723）已在运行，用来生成候选距离；冻结时还没有结果。Self Forcing 和 LongLive 的校准数据是预注册 27、28 的结果，已经公开。

## 规则（`select37.py`，冻结）
- 在校准过的距离 D > 20 中，取 [50, 60) 秒平均 α 最大、且冻结比例 ≤ 0.07 的那个，记为 D\*。
  - α 与以前相同：(end_p16 − end_m16) / (open_p16 − open_m16)，open_p16 − open_m16 < 4 L\* 时无定义；对场景取平均。
  - 冻结：一个视频 [50, 60) 秒的平均 motion < 1.5 灰度级；两个版本都计入。
  - 并列时取较小的 D。没有距离满足条件时，该模型没有 D\*，不进入研究 38、39。
- **重新对齐的距离**（sink 紧贴窗口之前）= 窗口 − 1：Self Forcing、CausVid、GAN 为 20，LongLive 为 11。

## 校准数据
- **Self Forcing DMD：** 预注册 27 的 17 个距离，24 个场景，种子 0（`results/corr/s27_out/s27_results.json`）。
- **LongLive：** 预注册 28 的 17 个距离，24 个场景，种子 0（`results/corr/s28_out/s28_results.json`）。
- **CausVid（autoregressive_checkpoint，3 步 1000,757,522，shift 8，不 warp）和 Self Forcing GAN：** 新做。
  - 候选距离：对各自探针的平均曲线用预注册 27 的规则（`s27_distances.select`，K = 14），再加距离 20 作参照。
  - 校准场景：rx00–rx03、gx00–gx03（预注册 06 的 24 个场景中的 8 个），L\* −16 / +16，通用提示词，种子 0，60 秒（243 个潜在帧）。
  - sink 3，窗口 21，`hold_gen_sinkpos.py`（Self-Forcing-b，冻结）。
  - 每个模型 × 场景一个任务，跑完所有距离（arch_guard），不写视频。任务清单由 `make_s37_tasks.py` 在探针结果出来后生成（`jobs/s37.jsonl`，16 个任务）。

## 已知结果（规则用在已有数据上）
- Self Forcing：D\* = 66（α 0.405，冻结 0）。距离 90（0.364）冻结 25%，448（0.37）冻结 12%，都被排除。
- LongLive：D\* = 448（α 0.356，冻结 4%）。距离 90（0.315）冻结 12%，被排除；其次是 147（0.287）。
- 这两个结果是在制定规则前就能看到的，所以它们只说明规则会选出什么，不构成检验。检验在研究 38。

## 预测（CausVid、GAN）
- 两个模型和 Self Forcing 是同一结构，探针曲线应和 Self Forcing 相关（LongLive 为 0.94），尖峰位置相同或接近。
- GAN 在 60 秒保留最多（原始位置 sink 3 时 0.79），预计 D\* 落在份额中等的距离（约 60–100），α 约 0.5。
- CausVid 的 sink 效果较弱（60 秒 0.23 对无 sink 0.04），预计 D\* 的 α 约 0.2–0.3；也可能没有距离同时满足保留多和不冻结，此时如实报告。

## 冻结文件（SHA-256）
```
5aded6f546a31a3d7c26ef49fbce8e9b2e0dfe78e4b65ee5dee1c4eb1d25580c  select37.py
73e081bcd6ba89d3dd17848aaf34b053bfaf72cfb79202151b2139dc9e91366f  make_s37_tasks.py
ea6a93de994e02811cfdc2cd406db6856fd84b6bc255ff2369b008837402f9ce  s27_distances.py
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
1d2d8dbece97de8b2a54af51d229db8aec41269707946387d14bda324cee8957  jobs/p37_probe.jsonl
```
- 规则已用在预注册 27、28 的结果上（`jobs/dstar_sfll.json`）：Self Forcing 66，LongLive 448。
