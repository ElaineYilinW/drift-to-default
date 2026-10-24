# 预注册 09：场景内容也会回归到提示词的默认值吗？（2026-10-24，生成前冻结）

## 背景
- 预注册 06/07 显示：亮度和色温会向提示词设定的默认值回归，sink 保留开头的一部分（α = w_s/(1−w_w)）。
- 这些都是全局色调统计。审稿人自然会问：**场景内容**（雪地、沙漠、森林……）是不是也一样？如果内容也回归，"回归到默认值"就是一般规律，而不只是色调漂移。
- 本研究用 CLIP 测量开头的场景类别在续写过程中保留多少，并用"通用城市提示词 vs 开头自己的类别提示词"作为配对操纵。

## 设计
- **类别（6 个）：** 雪地、沙漠、森林、海岸、山路、农田。提示词与预注册 06 的通用提示词同一模板，只替换地点和可见物（`prompts/content.jsonl`）。
- **开头：** Self Forcing DMD 用每个类别的提示词从零生成 5 秒视频（种子 0–15），取第 64 帧（4 秒）。CLIP ViT-L/14 在 7 个标签（6 个类别 + 城市）中分类；按种子顺序，取前 4 个"top-1 是自己类别且概率 ≥ 0.5"的帧作为开头（`content_openings.py`）。共 24 个开头。
- **续写：** 每个开头静止保持 9 帧（与之前相同），然后在两种提示词下续写：
  - mis：预注册 06 的通用城市提示词；
  - mat：开头自己的类别提示词。
- **配置（8 个）：** Self Forcing DMD sink 3 / 无 sink（60 秒，hold_gen_long）；LongLive sink 3 / 无 sink；CausVid；Self Forcing GAN；Self Forcing SiD；Rolling Forcing（各 31 秒）。种子 0。
- **默认值：** 每个配置用通用提示词从零生成 8 个视频（种子 0–7），时长同上。
- **执行：** 每个开头 × 模型家族的所有臂在同一个任务里运行，arch_guard 保证配对的臂在同一 GPU 架构上（`make_content_tasks.py`）。共 24 × 2 × 8 = 384 个续写视频 + 64 个从零开始的视频。
- **指标：**
  - 帧级类别分数 z = CLIP logit scale × (cos(帧, 类别标签) − cos(帧, 城市标签))，即该类别相对城市的 log-odds；嵌入取第 1–8 帧和第 9 帧之后每隔 4 帧（`clip_probe.py`）。
  - s(T)：开头之后 [T−5, T) 秒内 z 的均值；s_open：第 1–8 帧的均值；μ(T)：同一配置从零开始视频在同一窗口的类别分数均值。
  - 内容持续度 κ = (s(T) − μ(T)) / (s_open − μ(T))。s_open − μ < 2 的开头不进入 κ 分析。

## 假设（`analyze_content.py`，冻结；Holm 校正，共 3 个检验）
- **H1 提示词决定内容默认值：** Self Forcing DMD sink 3，T = 30 秒，s_mat − s_mis > 0。
- **H2 sink 保留内容（LongLive）：** T = 30 秒，κ(ll_s3, mis) − κ(ll_s0, mis) > 0。
- **H3 sink 保留内容（Self Forcing）：** T = 60 秒，κ(sf60_s3, mis) − κ(sf60_s0, mis) > 0。
- **检验方法：** 对每个开头算差值，做双侧单样本 t 检验；判定标准：Holm p < 0.05 且估计 > 0。95% CI 用 bootstrap（在类别内重抽开头，同时重抽从零开始的种子，10,000 次）。
- **次要分析：**
  - 每个配置 × 提示词在 10/20/30 秒（Self Forcing 加 40/50/60 秒）的 κ；
  - 8 个配置的 κ(mis, 30 秒) 与预注册 06 的亮度持续度 α_L(31 秒) 的 Spearman 相关（预测为正）；
  - 色调归一化后的 H1（每帧的 CIELAB 均值和标准差匹配到开头，排除"类别分数只是色调"的解释）；
  - 每个配置的 s_mat − s_mis；每个类别的 κ；7 个标签中开头类别的 top-1 准确率。
- **预测：** H1–H3 成立；所有配置 κ(mis) < κ(mat)；κ(mis, 30 秒) 在 0 到 0.7 之间。

## 冻结文件（SHA-256）
```
aba70706f039c7bb0836b28456a790c2b93fea7d83e440540f34cb0e44108a9f  analyze_content.py
002fe72cfc2c768d72e6f17f53228fb66664278ba12f6a6a1a0008dfc700d08f  clip_probe.py
cd6f9ec8b700a4bfccf42ecd3a8d758046bfa70c452f6b465dadaf6bd9cd48c9  content_openings.py
8b4770ce1e76d3efbdb0b9773c3869c367f0b3480263bd9412368c48acb69f4a  make_content_tasks.py
44f6ace3ebc86d89538354829c0183432c1f962733cddbfd8a4272d295f64607  arch_guard.py
cfb7068527b22670c611c8b9155b2570875e49c5625a84fe1f62a4ee3a7c38dc  prompts/content.jsonl
6fab80a4a5693d8c94ca18006759334c4b9c9cd8f7689fa5b8e5b6423d37ea8f  jobs/content_a.jsonl
d2910b7a2784dd477bb9ee50974d3f9361b4208f9e34321e996e602637f7cd10  jobs/content_c.jsonl
```
- 第二阶段任务清单 `jobs/content_b.jsonl` 由 `make_content_tasks.py b` 在开头选定后生成，其内容完全由冻结脚本和选出的开头决定。
