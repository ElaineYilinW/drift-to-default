# 预注册 39：不给开头时，固定在 D\* 的 sink 是否比重新对齐的 sink 颜色漂移更少、成像质量下降更少？（VBench 式指标，4 个模型 + Rolling Forcing 参照）（2026-11-02，生成前冻结）

## 背景
- 研究 37、38 测的是"开头保留多少"。审稿人会问：这对一般的长视频生成有没有用？本研究用文字生成的 60 秒视频（第一段由模型自己生成，并作为 sink），按 VBench 的做法分段评测质量，比较 4 种 sink 设置。
- 预注册 30：Self Forcing 不加 sink 时，60 秒结尾离起点 16.4 L\*，重新对齐的 sink 9.6，原始位置的 sink 3.8（但有突跳）。
- **数据状态：** 冻结时本研究没有生成任何视频。

## 设计
- **提示词：** `prompts/main.jsonl` 的 40 个（20 个 MovieGen、20 个驾驶）。**种子：** 13（预注册 30–36 用的是 10–12）。**时长：** 60 秒（243 个潜在帧）。不给开头（`--fresh`）。写视频。
- **模型和条件：** 与研究 38 相同的 4 个模型（有 D\* 的才参加），每个模型 4 个条件：
  - s0：不加 sink；
  - rr：sink 3 固定在重新对齐的距离（20，LongLive 11）；
  - held：sink 3 固定在 D\*；
  - abs：sink 3 在原始位置。
- **参照：** Rolling Forcing（发布的设置，`rf_gen_long.py`）。
- **执行：** 每个模型 × 提示词一个生成任务（arch_guard），由 `make_s39_tasks.py` 生成（Self Forcing、LongLive、Rolling Forcing 现在生成；CausVid、GAN 在研究 37 之后生成）。之后每个模型 × 提示词一个评测任务（`vbench_lite.py`）。

## 指标（`vbench_lite.py`，冻结）
- 每个视频去掉第一段（第 0–8 帧）后切成 10 秒的段，第 k 段 = 第 [9 + 160k, 9 + 160(k + 1)) 帧。每段：
  - **成像质量（imaging）：** MUSIQ（SPAQ 权重），每 8 帧取 1 帧，长边缩到 512（VBench 的做法）；
  - **美学质量（aesthetic）：** LAION 美学预测器 v1（CLIP ViT-L/14 嵌入上的线性头），每 8 帧取 1 帧，除以 10；
  - **主体一致性（subject）：** DINO ViT-B/16 的 CLS 特征，每 4 帧取 1 帧，段内用 VBench 的公式 (cos(f_i, f_1) + cos(f_i, f_{i−1})) / 2 取平均；
  - **背景一致性（background）：** 同样的公式，用 CLIP ViT-B/32；
  - **运动（flow）：** RAFT-large 光流，每 2 帧取 1 帧（8 fps），半分辨率，每对取幅值最大的 5% 的平均（VBench 动态程度在阈值化之前的分数）。
- **颜色偏移：** |[50, 60) 秒的平均 L\* − [0, 5) 秒的平均 L\*|（生成器记录的逐帧 L\*）。
- **成像质量变化：** 第 5 段 − 第 0 段的成像质量。

## 假设（`analyze_s39.py`，冻结；对所有参加的模型一起做 Holm 校正，最多 8 个检验）
- **B1_m：held 颜色漂移更少：** 偏移(rr) − 偏移(held) > 0。
- **B2_m：held 成像质量下降更少：** 成像质量变化(held) − 成像质量变化(rr) > 0。
- **检验方法：** 以提示词为单位的双侧单样本 t 检验，Holm p < 0.05 且估计 > 0 为成立；95% CI 用提示词 bootstrap（10,000 次）。

## 次要分析
- 每个模型和条件（以及 Rolling Forcing）各维度在第 0 段、第 5 段的平均值和变化，颜色偏移，[50, 60) 秒的 motion 和 flow，冻结比例；
- held 对 s0、对 abs 的同样比较，以及 flow；
- 驾驶提示词和 MovieGen 提示词分开的结果。

## 预测
- **B1：** Self Forcing 成立（rr 约 9.6，held 约 5–7 L\*）；LongLive 可能成立但差值小（它的偏移本来就小）；GAN、CausVid 方向相同。
- **B2：** 不确定。sink 让画面更接近起点，可能减少成像质量下降；但也可能让细节变糊。预计差值小（0–0.03）。
- **解读规则：**
  - B1 成立而 B2 不成立：固定距离减少颜色漂移，但不改善 VBench 的成像质量。论文据实写。
  - flow 明显低于 rr（次要分析）：held 的稳定部分来自运动减少，需要报告。
  - abs 在这些指标上可能和 held 一样好甚至更好；它的问题是突跳，这一点由研究 38 衡量。

## 冻结文件（SHA-256）
```
e36a2d383a3f6fe1af9bdf09692e7ef17d9e6c3f739a0cec98f8d12395647079  analyze_s39.py
3a40381cfc71b5e17b7a4c11813e06d8e5a8bd1162f35f58745e73a3e8532675  make_s39_tasks.py
3c0d5497abbd7662b4be15a76c7561a5955f62fdba004d31823be555fab3d637  vbench_lite.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
5fa88e36ae2c0cbfd71b6232f3c5618ca83ad12b4660b6d91523d145e77b9d9b  hold_gen_sinkpos.py
073f21f1476aea62a192de6aa71b00159c2af78a44bfa2304f025b7ed65821d7  ll_hold_gen_sinkpos.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
07c8f933393d8fd442f0df71b7cb74923ce1f9f5de6677b0fd370ef9b5562aa2  jobs/s39_sf_ll_rf.jsonl
aca4ef833e3064c9464ceb62ce02534d724fb9556c673ac651ac929331b2278c  jobs/s39_feat_sf_ll_rf.jsonl
db196bcf60cf2973e82153db47a8a57cbf6758b247a8baded314df6176bc2651  jobs/vblsmoke.jsonl  (smoke test only)
```
- 评测脚本在两个已有的 60 秒视频上做过冒烟测试（`jobs/vblsmoke.jsonl`）：每个视频得到 6 段，各维度数值在正常范围内，两个视频共 38 秒。
