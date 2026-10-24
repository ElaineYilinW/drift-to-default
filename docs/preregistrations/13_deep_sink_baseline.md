# 预注册 13：更深的 sink 作为校正的基线（2026-10-24，生成前冻结）

## 背景
- 内容感知校正（预注册 02）只和事后颜色匹配做过比较。事后匹配按构造会抹掉所有全局变化，是较弱的基线。
- Deep Forcing 提出的免训练做法是"更深的 sink"（把更多早期帧留作 sink）。按我们的递推模型，sink 越深，保留的开头越多（预注册 10 的 H1 检验持续度），所以它也应该减少漂移。但它通过注意力作用于全部内容，可能也会压住正常的亮度变化。
- 本研究在预注册 02 的确认集上，比较更深的 sink 与校正。

## 设计
- **数据集：** 预注册 02 的确认集：40 个 held-out 漂移提示词 × 4 个种子，8 个正常亮度变化的对照提示词 × 4 个种子；Self Forcing DMD，窗口 21，31 秒。
- **新分支：** sink 6（s6）和 sink 9（s9），不做校正（`correct_gen.py --sink 6|9 --method none`）。与原有的不校正（sink 3）和校正分支比较。
- **执行：** 与原确认集相同，固定在 A100/A800 上（这两种 GPU 输出逐位一致），与原有分支严格配对（`jobs/deepsink.jsonl`）。共 48 × 4 × 2 = 384 个视频。
- **指标与分析：** 与冻结的 `analyze_confirm.py` 完全相同（D、z_L、闪烁、对照集的正常变化保留比例），只扩展分支列表（`analyze_deepsink.py`）。

## 假设（`analyze_deepsink.py`，冻结）
- **主要假设（Holm 校正，共 2 个检验，单侧）：** D(s6) < D(不校正)；D(s9) < D(不校正)。判定：Holm p < 0.05 且 95% CI 上界 < 0。
- **次要分析：**
  - 对照集上正常亮度变化的保留比例（预注册 02 的标准是 ≥ 0.80）；
  - D(s6) − D(校正 β=1)、D(s9) − D(校正 β=1)（双侧，两个分支的混合模型）；
  - z_L；闪烁比例。
- **预测：** 两个主要假设成立；更深的 sink 保留的正常亮度变化少于校正。

## 冻结文件（SHA-256）
```
5a80004d1b767fdc62462d6447575917c43dd5289615bc3063453f642af5be80  analyze_deepsink.py
d079fc471d43c3988012d73cd2cb507281f6e9f4093acf767fe5bb2ae976617b  correct_gen.py
550605c5f50b1f9cc2b2721085ef248d1d7403fa9d3e6565c796d194b5a71bc0  jobs/deepsink.jsonl
dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124  prompts/main.jsonl
31eeca8883bb20785123a9b4bd7affa13a94d5cc8bd0c285641379015b3e3f8d  prompts/controls_confirm.jsonl
```
