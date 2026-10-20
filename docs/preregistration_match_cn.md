# 预注册：让提示词描述开头画面，能否减少漂移？（2026-10-19，生成开始前冻结）

## 背景与问题
Part 4 是探索性分析，发现 31 秒后画面亮度会回到提示词规定的默认水平：开头帧亮度偏差只剩约 22%。两个模型都是这样，斜率约 −0.7 到 −0.8。

由此得到一个可检验的预测：漂移中有很大一部分来自"提示词与开头画面不一致"。如果提示词本身就描述了开头画面（场景和光照），漂移应该明显减小；内容感知校正（ttc_s1）应能在此基础上进一步减小漂移。

本实验是 Part 4 的确认性检验。开头帧全部是新的，与之前用过的 40 张不重叠。

## 设计
- **开头帧（40 张，`prompts/match_openings.jsonl`，选择规则见 `jobs/select_openings.py`）**
  - 真实帧 nr00–19：取 NAVSIM navtest 中之前没用过的 127 段日志，用每段第 20 帧的 CAM_F0 图像。按亮度在最暗（15.6）到最亮（60.7）之间等距取 20 个目标值，每个目标取亮度最近的一段日志（`jobs/scan_logs.py`）。
  - 生成帧 ng00–19：Self Forcing 确认性实验中 20 个驾驶提示词、种子 1 的第 0 帧。之前的 self 开头帧用的是其他视频和种子 0。
  - 每张开头帧都固定 0.5 秒（9 帧），作为第一个 chunk，之后续写到 31 秒（501 帧）。
- **提示词（`prompts/matched.jsonl`，生成前冻结）**
  - G：之前真实开头实验用的通用提示词。
  - M：用 Qwen2.5-VL-3B-Instruct 对开头帧做贪心解码，生成描述（脚本 `caption_vlm.py`），输出 scene（场景）和 lighting（光照）两项，拼成："…driving {scene}. {lighting}. The camera moves forward steadily with realistic lighting and colors."
  - ML：通用的城市道路场景 + 描述中的 lighting 句子，用来单独看"只匹配光照"的效果。
  - 格式规则：scene 去掉句号、首字母小写；不以 on/through 开头时，前面加 "through "。lighting 首字母大写。这个规则在任何视频生成之前就已应用到全部 40 条，原始描述保存在 `caption` 字段。
- **分支（5 个）：** G、G+ttc、M、M+ttc、ML。ttc 统一为 at_step 1，β = 1。
- **模型（2 个）：**
  - Self Forcing：窗口 21 + sink 3，`correct_gen2.py`。
  - LongLive 1.0：窗口 12 + sink 3，`ll_gen.py`。
- **种子：** 0、1。共 40 × 5 × 2 × 2 = 800 个视频。
- **配对：** 同一模型、同一开头帧的 5 个分支和 2 个种子，在同一个任务内顺序运行，因此在同一块 GPU 上（`jobs/match_tasks.jsonl`）。

## 结局指标（相对开头帧）
- **主要指标：** |dL_end|，即最后 2 秒（32 帧）的平均 L\* 与开头帧（第 1–8 帧）平均 L\* 之差的绝对值。
- **次要指标：**
  - dE_end：Lab 均值之差的欧氏距离。
  - 有符号 dL_end。
  - 闪烁：jitter，以及 chunk 边界跳变。
  - 内容保持（retention）：最后 2 秒的 DINOv2 CLS 与开头帧 CLS 的余弦相似度，越高越好。

## 假设与检验（`analyze_match.py`，冻结）
- **主要假设（每个模型各检验一次，共 4 个检验，Holm 校正）**
  - H1：M 的 |dL_end| 小于 G。
  - H2：M+ttc 的 |dL_end| 小于 M。
- **检验方法：** 先在每张开头帧内对两个种子取平均，再算配对差，做双侧单样本 t 检验，同时给出开头帧聚类 bootstrap 95% CI（10000 次）。
- **判定标准：** Holm 校正后 p < 0.05 且估计值 < 0，则该假设成立。
- **整体结论的判定：**
  - H1 在两个模型上都成立 → "让提示词与开头画面一致能减少漂移"得到确认。
  - 只在一个模型上成立 → 结论限定于该模型。
  - H2 同理。
- **次要分析（只报告估计值和 CI，不做多重校正，不作为确认性结论）**
  - ML vs G：只匹配光照的效果。
  - G+ttc vs G。
  - M vs G+ttc：改提示词和校正哪个更有效。
  - M+ttc vs G。
  - 交互项：提示词 × 校正。
  - 按真实帧 / 生成帧分开报告 H1。
  - 各分支的回归斜率和零漂移亮度（预期 M 的斜率更接近 0）。
  - H1 的收益与开头帧到 G 默认亮度距离的关系。
  - dE、内容保持、闪烁比值。

## 预期与风险
- **预期：** 偏暗或偏亮、远离默认亮度的开头帧，从 M 中获益最大。
- **风险 1：** M 同时改变了场景描述，可能引入别的变化（比如更多运动），所以加入 ML 来分离光照的作用。
- **风险 2：** 描述里写"傍晚 / 夜晚"，可能反过来要求画面继续变暗。
- **风险 3：** NAVSIM 测试集里没有真正的夜景，最暗的真实帧是阴影和停车场。

## 冻结文件（SHA-256）
```
315fbf7d6e18c9b6002f8dcddd8c02e59be77087143e6d7a4e793cfd1aec22de  analyze_match.py
c4ea682a419fc3e4903e4f4539820813094ca6d0edabfd71480f01b84136cc10  caption_vlm.py
cee2d465d38345a2f487debc5983b8161dac02870653680ac212ec6ea28b41a2  jobs/select_openings.py
16403f81203011a5dd53bd7bcc06df30bf5404eef2c52cbbd409ebe3e22e2d17  prompts/match_openings.jsonl
7aac5cce3728a0f9dc4f6bce4bb965c8a0e568a202fa4de6f438bdebf30436e0  prompts/matched.jsonl
5f6e35ca194e4999d99271ab3ac9a531bdd364119da5703e1ec1f212c2bd6ba9  jobs/match_tasks.jsonl
7418c2f266fba61f4c2261c2f8a1f50245d4d8136dc8d83d67db58252ee1b81d  correct_gen2.py
b351d4b017fa51226488ae2d288b18f3b9cbbb54c362ef6dd1f006d376dd31f2  ll_gen.py
f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454  features.py
```
运行方式：`python analyze_match.py OUT SF=RUN_SF LL=RUN_LL`。
