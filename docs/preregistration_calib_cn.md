# 预注册：按实测亮度选光照描述，能否减少漂移？（2026-10-20，生成开始前冻结）

## 背景
匹配提示词实验（`docs/match_results_cn.md`）的结果是否定的：用 VLM 自由描述开头帧写提示词，没有可靠地减少亮度漂移。探索性分析找到的原因是：VLM 的光照用词与实际亮度对不上，它常把光照正常的画面说成"昏暗、清晨或傍晚"，提示词因此把画面往暗处拉。

由此得到可检验的假设：如果光照描述按开头帧的**实测亮度**来选，而不是让 VLM 自由写，漂移应当减小。

另外，晴天提示词的默认亮度（≈43）大约是文字能达到的上限，所以预期比它更亮的开头帧获益有限。

## 设计
- **开头帧、模型、种子：** 与匹配提示词实验完全相同。40 张开头帧，Self Forcing 和 LongLive，种子 0 和 1。
- **GPU 架构：** 每张开头帧固定在它之前 G 分支所用的 GPU 架构上，用 `gpu_only` 指定（A40；A100/A800；L40S/RTX 6000 Ada）。这样新分支和旧分支（G、M、M+ttc 等）严格配对。
  - 每个任务还会重跑 G 的种子 0 作为配对检查，预期与旧文件逐位相同。
- **光照分类规则（`make_calib_prompts.py`，生成前已应用）：**
  - 候选类别是 Part 4 的四种光照提示词。它们在 Self Forcing 上的零漂移亮度，来自更早那批独立的 40 张开头帧：night 16.0、dusk 19.1、overcast 33.6、sunny 42.9。
  - 每张开头帧取零漂移亮度离它的开头亮度最近的那一类。开头亮度用 Self Forcing G 分支实测的 L_open（两个种子的均值），两个模型用同一个值。
  - 分类结果：sunny 23、overcast 10、dusk 5、night 2。
- **新分支：**
  - CL：通用场景 + 校准光照，也就是该类 Part 4 提示词原文。
  - C：VLM 的场景短语（与 M 相同）+ 校准光照。
  - C+ttc：C 加上校正（第 1 步，β = 1）。
  - 共 40 × 2 模型 × (3 分支 × 2 种子 + 1 个检查视频) = 560 个视频。
- **已知局限：** C 里 VLM 的场景短语有时带有天气或时间词，比如 "on a wet, rainy night" 被分到 dusk、"under an overcast sky" 被分到 sunny，会和校准光照冲突。CL 没有这个问题，作为更干净的对照。

## 对匹配提示词实验数据的修复（与本实验一起执行）
- **问题：** 生成脚本会跳过已经存在的输出文件。任务被抢占后重跑时，是接着之前的进度做，而不是从头重做。因此 LongLive 上 ng16–ng19 这 4 张开头帧，前三个分支（G、G+ttc、M）在 A100 上生成，后两个分支（M+ttc、ML）在 A40 上生成，配对被破坏。
- **其他实验已检查：** 所有其他配对实验都在同一种 GPU 架构内，没有这个问题。
- **修复方法：** 把这 4 张开头帧的 ML 和 M+ttc 移到 `corr/match_ll_a40_split/`（保留），再在 A100/A800 上重新生成。
- **之后的处理：**
  - 用冻结的 `analyze_match.py` 重新分析，作为**敏感性分析**；原来的结果保留为"按执行"的结果，两者都报告。
  - 这个修复只影响 LongLive 的 ML 和 M+ttc，不影响匹配实验 H1（M vs G）的数据。

## 结局指标
与匹配实验相同。主要指标是 |dL_end|；次要指标是 dE_end、有符号 dL_end、内容保持、闪烁。

## 假设与检验（`analyze_calib.py`，冻结）
- **主要假设（每个模型各检验一次，共 4 个检验，Holm 校正）：**
  - H1：C 的 |dL_end| 小于 M。场景相同，只把 VLM 的光照描述换成按亮度校准的描述。
  - H2：C 的 |dL_end| 小于 G。
- **检验方法：** 每张开头帧先对两个种子取平均，再算配对差，做双侧单样本 t 检验，并给出开头帧聚类 bootstrap 95% CI。
- **判定标准：** Holm 校正后 p < 0.05 且估计值 < 0，则该假设成立。
- **次要分析（不做多重校正）：**
  - 配对差：CL − G，C − CL，C+ttc − C，C+ttc − G，C+ttc − M+ttc。
  - 内容保持：C − M（预期接近 0，因为场景相同），C − G。
  - 按光照类分组的 C − G 和 CL − G：night/dusk/overcast（17 张）与 sunny（23 张）分开。预期获益集中在前者。
  - 样本外检验 Part 4 的规律：用 Part 4 在 Self Forcing 上拟合的各提示词斜率和零漂移亮度，预测 CL 分支的 dL_end，报告 R²、相关和平均误差。
  - 闪烁比值。
  - 配对检查：G 种子 0 的重跑与旧文件逐位相同的比例。

## 冻结文件（SHA-256）
```
63adfed7ff451539ffec91cb7d03b829d290b86e3a0eb8391a57de0352bfbcaf  analyze_calib.py
b3a631746c25fda1a51558a9c89f6b0bf44dd8a40fed0bcc35bb45d77a21183c  make_calib_prompts.py
b08602d83481323f3b6d1bbee5e0c2fdd93f09987b7b2fe2d308a9c98a1ac704  prompts/calibrated.jsonl
860c8be4ad14e7cef8f003c75ed50310bf8056e3e8ce29ad202f078c98756d88  jobs/calib_tasks_a40.jsonl
887ce7db03ea44b9adefcde90a727e693d22c5739721d1079c960c5782681952  jobs/calib_tasks_a100.jsonl
6b512116ce7b54ba3e623428e9b66a97b0c798998fc0842ff0fed7474c967727  jobs/calib_tasks_l40s.jsonl
7418c2f266fba61f4c2261c2f8a1f50245d4d8136dc8d83d67db58252ee1b81d  correct_gen2.py
b351d4b017fa51226488ae2d288b18f3b9cbbb54c362ef6dd1f006d376dd31f2  ll_gen.py
f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454  features.py
```
运行方式：`python analyze_calib.py OUT results/corr/light_out/light_summary.json prompts/calibrated.jsonl SF=MATCH_SF,CALIB_SF LL=MATCH_LL,CALIB_LL`
