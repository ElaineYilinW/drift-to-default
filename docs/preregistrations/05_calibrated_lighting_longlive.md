# 预注册：LongLive 用自己的亮度标尺校准光照描述（2026-10-20，生成前冻结）

## 背景
上一个实验（`docs/calib_results_cn.md`）把光照描述按开头帧的实测亮度来选，用的是在 Self Forcing 上测出的亮度标尺。结果在 Self Forcing 上成立（|dL| −26%，Holm p = 0.013），在 LongLive 上不显著。用这把标尺去预测 LongLive 的结果，系统性偏高 +4.3 L\*。由此推测：标尺必须按模型单独测。

## 第一步（已完成，探索性）：测 LongLive 的标尺
- **做法：** 用 Part 4 的 40 张旧开头帧（real00–19、self00–19，与本实验的新开头帧互相独立），在 LongLive 上分别配 4 种光照提示词，每种 2 个种子，共 320 个视频。分析脚本为 `analyze_light.py`，结果在 `results/corr/ll_light_out/`。
- **LongLive 各提示词的零漂移亮度：**

  | 提示词 | 零漂移亮度 | 95% CI |
  |---|---|---|
  | night | 21.2 | [16.7, 23.8] |
  | dusk | 32.3 | [30.4, 34.4] |
  | overcast | 39.8 | [37.7, 41.7] |
  | sunny | 44.8 | [43.2, 46.6] |
  | 通用提示词（对照） | 39.9 | — |

  Self Forcing 上对应为：night 16.0、dusk 19.1、overcast 33.6、sunny 42.9。
- **光照项检验：** p ≈ 4×10⁻⁵¹。提示词决定默认亮度这一点在 LongLive 上复现了。LongLive 的范围更窄，dusk 尤其偏亮。

## 第二步（本预注册）
- **分类规则：** 沿用 `make_calib_prompts.py`，代码不变，只是把亮度标尺换成 LongLive 的。生成的提示词在 `prompts/calibrated_ll.jsonl`。
- **分类结果：** sunny 19、overcast 7、dusk 7、night 7。与按 Self Forcing 标尺的分类相比，有 16 张开头帧的类别变了：
  - overcast → dusk：7 张
  - dusk → night：5 张
  - sunny → overcast：4 张
- **生成与复用：**
  - 这 16 张在 LongLive 上重新生成 CL、C、C+ttc（各 2 个种子），外加 G 种子 0 作为配对检查。GPU 架构固定为它们原来 G 分支所用的架构（`jobs/calib2_tasks_*.jsonl`）。
  - 其余 24 张的提示词与上一个实验完全相同，而模型、种子、架构也相同，输出必然逐位相同，所以直接复制上一个实验的文件（`jobs/calib2_reuse.txt`）。
  - 所有文件放在 `corr/calib2_ll`。
- **主要假设**（用冻结的 `analyze_calib.py` 计算；只有 LongLive 一个模型，2 个检验，Holm 校正）：
  - H2（首要）：C 的 |dL_end| 小于 G。
  - H1：C 的 |dL_end| 小于 M。
- **判定标准：** Holm 校正后 p < 0.05 且估计值 < 0。
- **次要分析：**
  - 来自 `analyze_calib.py`：CL − G、C+ttc − G、内容保持；按光照类别分组；用 LongLive 自己的标尺做样本内预测（R²、平均误差）。
  - 来自 `analyze_calib_ll.py`：按 LongLive 标尺选的分支 vs 按 Self Forcing 标尺选的分支，分别在全部 40 张和类别改变的 16 张上比较。预期改进集中在这 16 张。
- **注意：** 第一步的标尺是在看过 LongLive 第一步数据之后确定的，但那些数据来自另一批开头帧，所以对本实验来说是独立的。

## 冻结文件（SHA-256）
```
63adfed7ff451539ffec91cb7d03b829d290b86e3a0eb8391a57de0352bfbcaf  analyze_calib.py
0d514f802890c6da95cb76c6f1c518ad9061be20c432c98bc9092da7c01c7eae  analyze_calib_ll.py
b3a631746c25fda1a51558a9c89f6b0bf44dd8a40fed0bcc35bb45d77a21183c  make_calib_prompts.py
51c31a69978fdc4e416fbbbb7c81489b992d5c9f7340cd0073e4a8b14248d19a  prompts/calibrated_ll.jsonl
c3f70f7516362bd7391615096286e7617aabe01c7c1d7a85b170a376067a7880  jobs/calib2_tasks_a100.jsonl
efffdffe7005ef7eaea14960bcb557523baf7449b4097fc0b9608c4c48114a96  jobs/calib2_tasks_a40.jsonl
ac7d3247fb467ce06087c74285d392005606374e39c53d00df1a67d0ee68d899  jobs/calib2_tasks_l40s.jsonl
30b2b6b0cb79aef4c026f96bff1764baecfd5d0e12ae5ecfe9c5f65476f3e182  jobs/calib2_reuse.txt
475eecb8cc125fcd792c24da60221b13bf8b735084d8443ad3a39e496ffc34c7  results/corr/ll_light_out/light_summary.json
```
运行方式：
- `python analyze_calib.py OUT results/corr/ll_light_out/light_summary.json prompts/calibrated_ll.jsonl LL=MATCH_LL,CALIB2_LL`
- `python analyze_calib_ll.py OUT CALIB_LL CALIB2_LL prompts/calibrated.jsonl prompts/calibrated_ll.jsonl`
