# 预注册 30：只用短片段训练的模型，从零生成时是否比用自己的长视频训练的模型离起点漂得更远？（40 个提示词，60 秒）（2026-10-28，生成前冻结）

## 背景
- 论文摘要和贡献列表里有一个探索性结论：漂移可以分成两部分，遗忘开头和"吸引子偏移"（默认值本身离同一提示词起点的距离）；用自己的长视频训练或微调的模型（LongLive、Rolling Forcing）几乎没有偏移，只用 5–10 秒片段训练的模型偏移 4–11 L\*。
- 这个结论的依据很薄（实验记录 18，探索性）：
  - 只有一个提示词（通用的城市行车提示词），4 个种子，31 秒；
  - Self Forcing DMD（sink 3）−6.0，无 sink −10.1；CausVid（sink 3）+11.2；LongLive（sink 3）+0.7，无 sink +2.3；Rolling Forcing +1.0。
  - 预注册 24 的从零生成（同一个提示词，4 个种子，60 秒）方向一致：Self Forcing 无 sink −12.7，重新旋转 sink −8.6，Rolling Forcing −0.5，LongLive（sink 3）−3.1。
- 本研究在 40 个提示词、3 个新种子上检验这一点。
- **数据状态：**
  - 冻结时本研究没有生成任何视频。
  - 这 40 个提示词以前用过：校正实验（研究 1–3）用它们生成过 Self Forcing（sink 3）和 LongLive（sink 3）的 31 秒视频（不同的生成器和种子），用来测校正前后的漂移。那些数据从来没有按"从零生成的偏移"在模型之间比较过，本研究不使用它们，冻结前也没有为这个问题去看它们。

## 设计
- **提示词：** `prompts/main.jsonl` 的 40 个提示词（20 个 MovieGen、20 个行车）。
- **生成：** 纯文本生成（`--fresh`，没有开头图像），60 秒（243 个潜在帧，969 帧，16 fps）。
- **种子：** 10、11、12（以前没用过）。
- **条件（生成器冻结，与预注册 24、29 相同）：**
  - sf_s0：Self Forcing DMD，无 sink（`hold_gen_long_rr.py --sink 0`）；
  - sf_rr：Self Forcing DMD，sink 3 重新旋转（`--sink 3 --rerotate`）；
  - cv_s0：CausVid，无 sink（同一生成器，CausVid 检查点和采样设置，与以前的 CausVid 实验相同）；
  - ll_s0：LongLive，无 sink（`ll_hold_gen_long.py --sink 0`）；
  - rf：Rolling Forcing（`rf_gen_long.py`）；
  - 次要：sf_s3（Self Forcing，原始位置的 sink 3）、ll_s3（LongLive，原始位置的 sink 3）。
- **为什么主要比较不用原始位置的 sink：** 原始位置的 sink 会周期性地突跳、复制开头（实验记录 31、32），这会让 60 秒处的"结尾"在某些时刻被拉回起点，与吸引子偏移混在一起。所以主要比较只用没有突跳的配置。
- **执行：**
  - 每个提示词 × 条件一个任务（3 个种子），共 280 个任务（`make_s30_tasks.py`，`jobs/s30.jsonl`）；
  - GPU 不限型号（≥ 40 GB，支持 bf16）；
  - 只为 drv_dawn_coast、mg0084 的种子 10 写视频文件；
  - 共 40 × 7 × 3 = 840 个视频。

## 指标
- 每个视频：
  - start = 第 1–8 帧的 L\* 均值（第一个块）；
  - end = 第 9 帧之后 [50, 60) 秒（第 809–968 帧）的 L\* 均值；
  - offset = end − start。
- 每个提示词 × 条件：offset 在种子上的均值（至少 2 个种子），以及它的绝对值 |offset|。

## 假设（`analyze_s30.py`，冻结；Holm 校正，共 3 个检验）
- **A1 Self Forcing 比 LongLive 漂得远（都无 sink）：** |offset(sf_s0)| − |offset(ll_s0)| > 0。
- **A2 CausVid 比 LongLive 漂得远（都无 sink）：** |offset(cv_s0)| − |offset(ll_s0)| > 0。
- **A3 Self Forcing（重新旋转的 sink）比 Rolling Forcing 漂得远：** |offset(sf_rr)| − |offset(rf)| > 0。
- **检验方法：** 按提示词配对（两个条件都有值的提示词），双侧单样本 t 检验，Holm p < 0.05 且估计 > 0 为成立；95% CI 用提示词 bootstrap（10,000 次）。
- **说明：**
  - A1、A2 是最干净的比较：三个条件都没有 sink，开头都会被忘掉（LongLive 无 sink 在 31 秒的 α = 0.02）。
  - A3 两边都把开头留在上下文里，但留下的份额不同（Rolling Forcing 约 0.3，重新旋转的 sink 约 0.16），所以 Rolling Forcing 的偏移会因此机械地小一些（约 15%）。A3 支持这个结论，但不如 A1、A2 干净。
- **把握度：** 不同提示词之间差值的标准差未知。若平均差值为 5 L\*、标准差 5 L\*（d = 1），40 个提示词在 0.017 水平上的把握度 > 0.99；若平均差值只有 2 L\*（d = 0.4），约 0.6。

## 次要分析
- 每个条件：
  - 带符号的平均 offset、平均 |offset|；
  - 去掉种子噪声的均方根偏移：每个提示词对种子两两相乘取平均（独立种子时其期望为真实偏移的平方），对提示词平均后取正部的平方根；
  - 种子间标准差的平均。
- A1–A3 改用去掉种子噪声的平方偏移；
- A1–A3 在 31 秒（end 取 [29, 31) 秒，对应实验记录 18 的定义）；
- A1–A3 分别在 MovieGen 和行车提示词上；
- sf_s3 与 ll_s3 的比较（原始位置的 sink，即两个模型的默认设置）；
- 每对条件中前者 |offset| 更大的提示词比例；
- offset 的 1 秒曲线；
- 坍塌计数（1 秒平均 L\* 低于 3 或高于 97 的视频数）。

## 预测
- **A1、A2 成立：** 差值约 4–8 L\*。
- **A3 倾向于成立，** 差值约 3–7 L\*。
- 不确定之处：
  - MovieGen 提示词有很多镜头运动和场景变化（推近、拉远），内容本身就会改变亮度，会增加所有条件的噪声；
  - 只用一个提示词时 Self Forcing 变暗、CausVid 变亮，在 40 个提示词上方向可能因提示词而异，所以检验用绝对值。
- **解读规则：**
  - 若 A1、A2 都成立：在 40 个提示词上，只用短片段训练的模型从零生成时离起点漂得比 LongLive 远，摘要里"偏移在用自己的长视频训练的模型里很小"可以保留（注明是 LongLive 的结果，并给出提示词数）。若 A3 也成立，Rolling Forcing 一并保留。
  - 若 A1 或 A2 不成立：漂移的两部分分解仍然成立（这是定义），但"长视频训练让偏移变小"要从摘要里删掉或改成只对某一类提示词成立。
  - 若 A3 不成立而 A1、A2 成立：这一结论只对 LongLive 成立，Rolling Forcing 的说法要删掉。
  - 无论结果如何，论文都会说明这些模型除训练时长外还有其他差别（数据、蒸馏方式、窗口），所以结论是相关性的。

## 冻结文件（SHA-256）
```
8c48f2f67f2db223db9f6e14c742bc6cc627f24367adc478926da9af8b259317  analyze_s30.py
e7cd9901615527e5855502b0c9e3d38c719d22f1acffe3c36da5b2d8b6bb08df  make_s30_tasks.py
6d586aeb19f779f0d431a55915dee13bc7675a83d00ecf010992a4a8548dc94c  jobs/s30.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
8723343defd6974f6d6f8a7c7e3d971272451882bc15952a4c70427d959a6a7e  ll_hold_gen_long.py
9b58613c2ed1f06862b9f9b5290e5bfd9a52a9c6cc147fc58409d26628bb2599  rf_gen_long.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
```
- 分析脚本已在合成数据上检查过（已知偏移 −10、−8、+10、+2、+1 能正确恢复；缺失的种子和提示词能正确处理）。
