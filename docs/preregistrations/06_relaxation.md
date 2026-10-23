# 预注册：长时程自回归生成的"弛豫"——开头保留多少、往哪里去、锚定和校正如何改变它（2026-10-23，生成开始前冻结）

## 背景与理论
已有结果：漂移是回归到提示词决定的默认水平。在已有数据上，探索性分析显示亮度、a\*、b\*、饱和度、清晰度、对比度都有同样的回归（开头值与变化量相关 −0.8～−0.95，两个模型都如此）。持续度随时间先快速下降，再在约 0.3–0.4 附近趋平。

在 sink 3 的设置下，被固定的开头 chunk 本身就是 sink。由此提出可检验的机制模型：

  长时程状态 ≈ α·开头 + (1 − α)·μ_prompt

- α：开头被保留的份额。按理论，sink 把开头永久留在上下文里，因此 α > 0；没有 sink 时，开头约 5 秒后就离开窗口，α 应接近 0。
- μ_prompt：提示词决定的默认水平。

这个模型把之前所有结果统一起来：sink 和内容感知校正是在提高 α；校准过的提示词是让 μ_prompt 靠近开头。

之前的证据都是观察性的：比较的是不同场景的开头亮度，而暗场景和亮场景在内容上也不同。本实验在**同一场景内**直接操纵开头的亮度和色温，得到干预性的估计。

## 设计
- **场景：** 24 个基础场景（`select_relax_openings.py`，规则在生成前固定）。
  - 12 个真实场景：NAVSIM 之前未用过的日志，开头亮度在 35–50 之间等距选取。
  - 12 个生成场景：Self Forcing 确认实验中驾驶提示词种子 2 的第 0 帧，取亮度最接近 42 的 12 个。
- **开头版本（每个场景 7 个）：** 在 CIELAB 中对图像做受控平移，再送入 VAE 编码。
  - L\*：−16、−8、0、+8、+16
  - b\*：−12、+12（L\* 不变）
- **提示词：** 全部使用之前真实开头实验的通用驾驶提示词。种子 0；同一场景同一配置的 7 个版本在同一个任务、同一块 GPU 上生成。
- **配置（模型 × 上下文策略）：**
  - sf_s3 / sf_s0 / sf_ttc：Self Forcing DMD，窗口 21，sink 3 / sink 0 / sink 3 加校正（第 1 步，β = 1）。三者在同一个任务中运行。
  - sid_s3、sidv2_s3、gan_s3、sf10s_s3：Self Forcing 发布的其他训练目标（SiD、SiD v2、GAN、10 秒训练），窗口 21，sink 3。ODE 回归初始化的学生模型在冒烟测试中开头之后即坍缩为模糊画面，其颜色统计反映的是坍缩而不是漂移，因此在冻结前剔除，用 SiD v2 代替。
  - cv_s3：CausVid（autoregressive_checkpoint，3 步 [1000, 757, 522]，shift 8，不做 warp；冒烟测试画面正常），窗口 21，sink 3。
  - ll_s3 / ll_s0：LongLive，窗口 12，sink 3 / sink 0。两者在同一个任务中运行。
  - rf：Rolling Forcing（锚点固定为开头 block）。
  - sky：SkyReels-V2-DF 1.3B（diffusion forcing，24 fps，544p，30 步，CFG 6）。冒烟测试中 8 秒视频需 13 分钟，所以只做缩减设计：每隔一个场景取 1 个，共 12 个场景；只用 L\* −16、0、+16 三个版本；时长 20 秒。
- **从零开始的对照：** 除 sky 外的每个配置，都用通用提示词、4 个种子，不固定开头，生成 31 秒。用来比较默认水平与从零开始的终点水平。
- **长时程子集（次要）：** sf_s3、sf_s0、ll_s3 三个配置 × 6 个场景 × L\* ±16，生成 120 秒。

## 执行细节
- 偶数场景固定在 A40，奇数场景固定在 L40S / RTX 6000 Ada（同一种架构）；同一场景同一配置的所有版本在同一个任务内运行，被抢占后续跑也不会换架构。任务清单由 `make_relax_tasks.py` 生成（`jobs/relax_a40.jsonl`、`jobs/relax_l40s.jsonl`）。
- `hold_gen.py` 在不加扰动、不换 checkpoint 时，与之前冻结的 `correct_gen2.py` 输出逐位相同（冒烟测试已核对）。

## 指标
- 开头值：被固定帧的均值（不含第 0 帧）。终点值：最后 2 秒的均值。属性为 L\*（主要）、b\*（主要）、a\*、饱和度。
- 持续度 α：同一场景内，终点值对开头值做 OLS，取斜率；再在场景间取平均。
  - α_L 用 5 个 L\* 版本估计。
  - α_b 用 b\* 的 −12、0、+12 三个版本估计。
- 默认水平：同一场景这条直线上"终点 = 开头"的那个点。
- 时间进程：每秒算一次 α(t)，再拟合 α(t) = α∞ + (1 − α∞)·exp(−t/τ)。

## 假设（主要，Holm 校正共 4 个检验）
- H1：α_L(sf_s0) < α_L(sf_s3)。sink 把开头留在上下文里。
- H2：α_L(ll_s0) < α_L(ll_s3)。在第二个模型上复现 H1。
- H3：α_L(sf_ttc) > α_L(sf_s3)。内容感知校正提高保留份额。
- H4：α_b(sf_s0) < α_b(sf_s3)。H1 推广到色温。

**检验方法：** 每个场景算出两个配置的斜率差，做双侧单样本 t 检验，并用场景 bootstrap 给出 95% CI。判定标准：Holm p < 0.05，且方向与预期一致。

## 次要分析（只报告估计和 CI）
- 每个配置的 α_L、α_b、α_a、α_sat 和默认水平；α(t) 曲线，以及半衰期 τ·ln2 和 α∞。
- 不同训练目标和模型之间的比较（DMD、SiD、SiD v2、GAN、10s、CausVid、LongLive、Rolling Forcing、SkyReels）。sky 的 α 在 20 秒处评估，与其他模型在 20 秒处的 α(t) 对照。
- 默认水平与从零开始终点水平的比较。
- 没有 sink 时，31 秒的 α_L 是否小于 0.2（接近完全遗忘）。
- 长时程子集在 30、60、90、120 秒的 α_L。
- 样本外定量检验：用本实验估计的 α_L(sf_s3)，加上 Part 4 的各提示词默认水平，预测 Part 6 校准光照实验中各开头帧的结果（这些数据已有），报告 R² 和相关。

## 冻结文件（SHA-256）
```
073eca48a63ed86491129ac946e1771c545209053fa3712fe6f90103563c5b47  analyze_relax.py
4f14ab0f30365c1f75c004e7b676712d3516010d864531d0663e248b140ce1a4  hold_gen.py
698c02be1ab51f4e788561a98011c43a847ab54c03ec8caea658ff61c5c2e36b  ll_hold_gen.py
d0e7f1b202ab1914e7ba07523efe043a069e8be38f9073017fa2400dc3d87e36  rf_gen.py
2c8feb5d4aff4ed7a68c3361f783e7076dceff3add730c5c037648f8015e9743  sky_gen.py
8f2195531cda19f49fc7a77254632b2e394119fc0bc3689ea5f9cd3c11b4bc4b  select_relax_openings.py
2bdc44089f01b92775ea17528ea869b000c0075d2b3a21bb66ae0442938f6fec  make_relax_tasks.py
7fc02aae9457e378d1d4be87d849a118302f2dff5e3cad6cafb08dd8df9dbd2f  prompts/relax.jsonl
7bb4d259098a580e86235ec79818ec4a2d123ae5b9871ca005e7cdfc6e0e7974  jobs/relax_a40.jsonl
20ad6780f704b191d19d81ed9ea7e326e152d30dd5c5395289a87bfc4abff26b  jobs/relax_l40s.jsonl
042bf0062da74964db9447fb21ab1151f0437dbf47fb9afebbce7daaf45c77ff  stats_util.py
```
运行方式：`python analyze_relax.py RELAX_DIR OUT`（RELAX_DIR 为 corr/relax 的统计文件）。
