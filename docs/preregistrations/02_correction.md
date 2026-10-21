# 预注册：参考内容的变暗校正（确认性实验，2026-10-19）

写于确认性实验生成之前。方法、参数和判定标准**全部来自开发集**（预实验的 10 个 prompt + 6 个开发用对照，见 `docs/correction_dev_cn.md`）。确认性实验使用**未参与开发**的 40 个正式 prompt 和 8 个新写的对照。分析脚本 `analyze_confirm.py` 和本文件同时冻结，SHA-256 记录在文末。

## 1. 方法
- 基础设置：Self Forcing DMD（Wan2.1-T2V-1.3B），窗口 21 个潜帧 + 锚定 3 帧（官方 sink 实现，不重新对齐位置编码），501 帧。
- 校正（`correct_gen.py --method ttc --at_step 1`）：从第 2 段开始，每段在去噪第 1 步（输入时间步 937.5）时，用只含锚定帧的临时 KV cache 对同一带噪输入再做一次前向，得到 x0ᶜ。把 x0ᶜ 和实际预测 x0 之间每个潜帧 16 个通道的空间均值差，乘以 β 加到 x0 上，然后照常加噪进入下一步。不训练，主 KV cache 和随机数序列都不变。
- 三个比较组：`none`（不校正）、`ttc_s1`（β = 1）、`ttc_s1_b15`（β = 1.5）。
- 配对：同一 prompt 的所有组和所有 seed，都在同一 GPU 架构上生成（A100/A800，已验证二者逐位一致）。所以在第一次校正之前，各组的画面完全相同。

## 2. 数据
- **漂移集**：`prompts/main.jsonl` 中的 40 个 prompt（20 个 MovieGen + 20 个驾驶），seed 0–3。参照分布沿用正式实验的参照视频（seed 100–109）。
- **对照集**：`prompts/controls_confirm.jsonl` 中的 8 个新写的亮度转换 prompt（5 个变暗、3 个变亮），seed 0–3。

## 3. 指标（终点为第 29–31 秒）
- **D**：颜色漂移，与正式实验的定义相同。
- **K**：DINOv2 分布漂移，与正式实验的定义相同。
- **z_L**：带符号的亮度 z 分数。
- **闪烁**：逐帧亮度二阶差分的 SD（抖动），以及段边界亮度跳变的均值。报告为相对不校正组的倍数，只做描述。
- **对照保留比例**：（校正组首末亮度差）÷（不校正组首末亮度差），先在每个 prompt 内对 seed 取中位数，再对对照取中位数。**如果某个对照在不校正组里的平均 |首末亮度差| < 5 L，说明模型根本没有产生转换，这个对照排除**（规则来自开发阶段的"黎明"对照）。

## 4. 检验
模型：y ~ 组别 + (1 | prompt) + (1 | prompt:seed)，REML，每个校正组分别与 `none` 比较。

- **主要假设族**（Holm 校正，2 个检验）：
  - H1a：`ttc_s1` 使 D 降低；
  - H1b：`ttc_s1_b15` 使 D 降低。
  - 单侧检验。判定为"支持"需要 Holm 校正后 p < 0.05，**且** 95% CI 上界 < 0。
- **次要假设族**（Holm 校正，2 个检验）：
  - H2a、H2b：z_L 升高（变暗减轻），单侧，判定方式同上。
- **结构非劣效**（不做假设检验）：ΔK 的 95% CI 上界 < +0.30，就判定为"不伤害结构"。+0.30 约为开发集中不校正组 K 值的 12%。
- **正常变化保留**（描述性判定）：在有效对照上，保留比例的中位数 ≥ 0.80，就判定为"基本保留正常变化"。
- **闪烁**：只描述，不做判定。开发集中 β = 1.5 时段边界跳变约为 3.3 倍，这是已知的代价。
- **稳健性**：按 prompt 聚类 bootstrap 的 CI；驾驶和 MovieGen 分开报告。

## 5. 偏离
任何改动都在最终报告中列出并说明理由。

## 冻结记录
- 冻结时间：2026-10-19 00:28（本地时间），确认性实验尚未生成。
- `docs/preregistration_correction_cn.md` sha256 c98d2dd50f1b543471992af7898e028da028dc247a02e95ec6ed3155d815fa47（本行追加之前的内容）
- `analyze_confirm.py` sha256 a5363d5c9b3cd484670f27023a7daf5e21c153e95edaf36db5a4500ba3a9202e
- `correct_gen.py` sha256 d079fc471d43c3988012d73cd2cb507281f6e9f4093acf767fe5bb2ae976617b
- `features.py` sha256 f6c15dba0bc97f0ba39b4a75c4c07c5f7f489eb9172d065f1f83d50434a95454
- `stats_util.py` sha256 042bf0062da74964db9447fb21ab1151f0437dbf47fb9afebbce7daaf45c77ff
- `jobs/patch_sf.py` sha256 e27d487e2fe816baa78961a33a600f15341bfdc5f27505ecafacf6228dffd897
- `prompts/main.jsonl` sha256 dc5d710e3e9f048e10c64dfabe9074559e67e8c794f848f163544451d1ab1124
- `prompts/controls_confirm.jsonl` sha256 31eeca8883bb20785123a9b4bd7affa13a94d5cc8bd0c285641379015b3e3f8d
