# 预注册 32：LongLive 的吸引子偏移小，是不是因为它的推理窗口更小？（Self Forcing 用 12 帧窗口，40 个提示词，60 秒）（2026-10-29，生成前冻结）

## 背景
- 预注册 30 检验"只用短片段训练的模型从零生成时比用自己的长视频训练的模型离起点漂得更远"（A1：Self Forcing 无 sink 对 LongLive 无 sink）。
- 但这两个模型的推理设置不同：
  - Self Forcing 的键值窗口是 21 个潜在帧；
  - LongLive 是 12 个。
- 所以即使 A1 成立，偏移小也可能来自窗口小，而不是来自 LongLive 在自己的长视频上的训练。
- 本研究让 Self Forcing DMD 用 LongLive 的窗口（12 个潜在帧），在研究 30 的同样 40 个提示词和种子上从零生成，与研究 30 的 LongLive 视频比较。
  - 两个模型的结构（Wan2.1-1.3B）、去噪步数（4 步）和窗口都相同；
  - LongLive 是在 Self Forcing 基础上用自己的长视频继续训练的，所以剩下的主要差别是这段训练（以及它带来的数据和流程差别）。
- **数据状态：**
  - 冻结时，研究 30 已生成约一半的视频，但它的结果还没有被读取或分析。本研究在读取研究 30 的结果之前冻结。
  - 研究 1（实验记录 02）用 gen.py 在同样 40 个提示词、种子 0–3 上生成过 Self Forcing 窗口 12 和 21 的 31 秒视频。记录 02 报告过最后 2 秒相对 fresh 5 秒视频的标准化亮度 z_L：
    - 无 sink：窗口 12 为 −1.81，窗口 21 为 −3.55；
    - sink 3：分别为 −1.97、−2.01。
  - 这些数字在冻结时是已知的，说明 Self Forcing 在两种窗口下都变暗。它们不是起点到终点的偏移，也没有和 LongLive 比较过；本研究不使用它们。

## 设计
- **提示词、长度、种子：** 与研究 30 相同。
  - `prompts/main.jsonl` 的 40 个提示词（20 个 MovieGen、20 个行车）；
  - 纯文本生成（`--fresh`），60 秒（243 个潜在帧）；
  - 种子 10、11、12。
- **新条件（生成器 `hold_gen_long_rr.py`，与研究 30 的 sf_s0 相同，只改窗口）：**
  - sf12_s0：Self Forcing DMD，窗口 12，无 sink（`--window 12 --sink 0`）；
  - sf12_s3：Self Forcing DMD，窗口 12，原始位置的 sink 3（`--window 12 --sink 3`），即 LongLive 的默认设置。
- **比较对象（研究 30 的视频，不重新生成）：**
  - ll_s0：LongLive，无 sink；
  - ll_s3：LongLive，sink 3；
  - sf_s0、sf_s3：Self Forcing，窗口 21。
- **执行：**
  - 每个提示词 × 条件一个任务（3 个种子），共 80 个任务（`make_s32_tasks.py`，`jobs/s32.jsonl`）；
  - GPU 不限型号（≥ 40 GB，bf16）。不同模型之间的比较本来就不是逐视频配对的，按提示词配对。
  - 只为 drv_dawn_coast、mg0084 的种子 10 写视频文件；
  - 共 40 × 2 × 3 = 240 个视频。

## 指标
与研究 30 相同：
- start = 第 1–8 帧的 L\* 均值；
- end = 第 9 帧之后 [50, 60) 秒的 L\* 均值；
- offset = end − start；
- 每个提示词 × 条件：offset 在种子上的均值（至少 2 个种子）及其绝对值。

## 假设（`analyze_s32.py`，冻结；Holm 校正，共 2 个检验）
- **W1 窗口相同、都无 sink 时，Self Forcing 比 LongLive 漂得远：** |offset(sf12_s0)| − |offset(ll_s0)| > 0。
- **W2 窗口相同、都是原始位置的 sink 3 时（LongLive 的默认设置），同上：** |offset(sf12_s3)| − |offset(ll_s3)| > 0。
- **检验方法：** 按提示词配对，双侧单样本 t 检验，Holm p < 0.05 且估计 > 0 为成立；95% CI 用提示词 bootstrap（10,000 次）。

## 次要分析
- 窗口本身在 Self Forcing 里的作用（双侧，估计和 CI）：
  - |offset(sf12_s0)| − |offset(sf_s0)|；
  - |offset(sf12_s3)| − |offset(sf_s3)|。
- 每个条件：带符号的平均 offset、平均 |offset|、去掉种子噪声的均方根偏移、种子间标准差的平均；
- W1、W2 的三种变体：
  - 改用去掉种子噪声的平方偏移；
  - 改在 31 秒（[29, 31) 秒）；
  - 分 MovieGen 和行车提示词；
- offset 的 1 秒曲线；
- 坍塌计数。

## 预测
- **W1 成立：** 差值约 4–8 L\*。根据研究 1，Self Forcing 窗口 12 无 sink 时也明显变暗。
- **W2 倾向于成立，** 差值较小（约 2–5 L\*）。原始位置的 sink 会周期性地把画面拉回开头，两边的偏移都会变小。
- **窗口本身的作用：** 预计在 ±3 L\* 之内。
- **解读规则：**
  - 若 W1 成立：LongLive 偏移小不能用窗口小来解释。论文写"在相同的结构、步数和窗口下，用自己的长视频继续训练的 LongLive 偏移更小"，仍注明这是一个模型对的比较。
  - 若 W1 不成立：
    - 窗口可能是原因之一。论文里"长视频训练让偏移变小"要改成"LongLive 和 Rolling Forcing 偏移小，原因可能是训练，也可能是推理设置"，并从摘要里删去训练这一解释。
    - 若研究 30 的 A1 成立而 W1 不成立，要同时报告两者。
  - W2 只作补充：它比较的是两个模型的默认设置。

## 冻结文件（SHA-256）
```
65bb7450f54916d1417e94dff521908cba259a59586ac868a8684473ddfb679f  analyze_s32.py
937449383b3bb9f9ddd14ff7063c933e7e44d1e2c2bce01ef1fdfd27e521c8ff  make_s32_tasks.py
30e3cfe47d78c89f024b59838f2ce115e9c144de0db55a7ac84e5a05306462fa  jobs/s32.jsonl
f7b6fe4808db786ec38e47cef560870b355f1f92c401361649db9a8c12dee718  hold_gen_long_rr.py
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
```
- 分析脚本已在合成数据上检查过：已知偏移 −10、−6、+2、+1、−8、−5 能正确恢复；缺种子的提示词会被正确去掉。
