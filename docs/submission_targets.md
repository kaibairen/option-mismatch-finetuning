# 投稿目标与实验门槛

状态：2026-09-29 第二轮。会议与期刊按下面第 0 节锁定。历史数字仍以 `reports/` 为准。

## 0. 这一轮锁定的两个出口

材料仍读不到本地的 NLPCC `main.tex`。方法名按实验目标里的 DECORD 实现：Diligence-guided End Commitment for Option Representation Drift。训练无关的 DECORD 和微调方案 DECORD-FT 都必须在同一套 AQUA-RAT test 上，字母准确率至少高于 Base、SFT、标准 DPO 里最强的一个 **1 个百分点**。选择发生在 dev 上，test 只报一次。

### 0.1 CCF 会议：NAACL 2027（CCF-B）

依据是 [NAACL 2027 main conference CFP](https://2027.naacl.org/calls/main_conference_papers/) 和 [ARR dates](http://aclrollingreview.org/dates)。2026-08 的 CCF 截稿清单仍把 NAACL 标为 B。CCF 第七版不把 Findings、workshop、short paper 算进目录，所以目标是 main long paper。

| 项 | 要求 |
| --- | --- |
| 投稿 | ARR 2026 年 10 月轮，截稿 **2026-10-12** 23:59 UTC-12。承诺 NAACL 的截止日期 **2026-12-23**。录用通知 2027-02-10。会议 2027-06-01 至 06-05，旧金山。 |
| 篇幅 | 长文 8 页正文，参考文献和附录不限。录用后可加 1 页到 9 页。 |
| 评审 | 双盲。先在 ARR 审，再承诺到 NAACL。与 COLING 2027 同一轮，一篇稿只能承诺一个主会。 |
| 内容 | 必须是实质性的、未发表的 NLP 贡献。CFP 点名了 Interpretability and Analysis、Inference-Time Methods、Reasoning、Safety and Alignment，和 DECORD 同题。只报一个小模型上 +1 个点、没有对照和失败分析，过不了「substantial contribution」。 |
| 必备部分 | Limitations 专节；和已有工作的直接比较；负责的 NLP 研究清单；不能一稿多投。 |
| 不接受 | 在 NAACL 审稿期间同时投 TNNLS。期刊版必须等会议流程结束，并且相对会议稿有实质扩展。 |

10 月 12 日之前如果 test 门禁还没过，这一轮不投 NAACL。稿仍按这 8 页长文的结构写，以免为了赶上截稿把负结果送审。

### 0.2 IEEE 期刊：TNNLS（CCF-B，公开目录中的 JCR Q1 / 2025 中科院计算机科学 1 区 TOP）

依据是 [IEEE CIS TNNLS Information for Authors](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls)。不选 KBS：KBS 不是 IEEE，而且在 CCF 目录里通常是 C 类。TNNLS 同时满足「IEEE」和「CCF-B」。投稿前用学校账号打开的 2025 分区表再核一次大类分区。

| 项 | 要求 |
| --- | --- |
| 范围 | 神经网络与学习系统的理论、设计或应用。全文要有可归档的新贡献，审稿人会拒增量式改进。 |
| 体裁 | Full Paper。双盲，至少两位审稿人。IEEE 双栏。摘要必须包含主要结果。4–5 个关键词。 |
| 篇幅 | 建议不超过 10 个印刷页。超过 10 页有强制超页费，绝对上限 15 页（不含补充材料）。 |
| 独占性 | 不能同时在别处审。已经发表的会议稿必须先做实质扩展才能投。 |
| 复现 | 算法、损失、基线、数据划分、种子要能让别人重跑。AI 生成的文字要在致谢里披露并引用系统。 |
| 作者 | 每位作者需要 ORCID。 |

和 NAACL 共用的内容门槛：问题定义、DECORD 与 DECORD-FT 的公式、Base / SFT / 带参考模型的 DPO 对照、dev 上选定的超参、test 上的准确率和错配率、去掉字母间隔项和 \(\alpha=0\) 的消融、局限（单模型尺度、单一数据集、+1 个百分点的统计不确定性）。

### 0.3 实验计划

1. 训练无关 **DECORD**：贪心生成后，在 `Final answer:` 边界上对 A–E 取下一个 token 的对数概率，再加上或减去 \(\alpha\)，奖罚「与 Phase II 数值所对应字母是否一致」。\(\alpha\in\{0,0.5,1,2,4\}\) 在 dev 上选，\(\alpha=0\) 是纯概率消融。
2. **SFT**：同一训练题上的均匀 token NLL。这是对照，不是我们的方法。
3. **DPO**：从该 SFT 初始化，冻结参考模型就是这个 SFT，偏好对的 chosen 是金标准解答，rejected 是 SFT 自己生成的错误收尾。损失是标准 DPO，含 \(\log\pi_{\mathrm{ref}}\)。
4. **DECORD-FT**：答案行 4 倍权重的 SFT，加上 gold 字母相对其他字母的 logit 间隔；再用同一批偏好对做带间隔项的 DPO。dev 上在 DECORD-SFT 和 DECORD-DPO 之间选一个作为微调方案，test 只评被选中的那个。
5. 机器是 RTX 2080 Ti 11GB。模型用 Qwen2.5-0.5B-Instruct，权重放在 `/root/autodl-tmp`。先跑 16 条 smoke，再跑 train 400、dev 60、AQUA test 全量。

出门条件写在 `scripts/experiments/decord_gate.py`：`decord_beats` 和 `decord_ft_beats` 同时为真才算这一轮完成。做不到就改 \(\alpha\) 的用法、采样条数或损失权重后再跑，不改 test 划分。

材料边界：本环境读不到 `/Users/yaosiyi/Downloads/NLPCC/main.tex`。下面的论文诊断依据是仓库里的实验设计、协议实现和 V100 冻结结果。若 tex 里还有仓库没有的主张，那些主张尚未被这些数字支撑。

## 1. 仓库现在实际证明了什么

科学问题被收成一句话：单步选择题里，Phase II 的中间演算仍沿着「严谨」方向，Phase III 把数值对到 A–E 时这个方向塌掉；塌缩是局部激活漂移，所以可以用只约束 Phase III 的后训练把它拉回来，并降低选项错配。

已经跑完的三层证据：

| 阶段 | 设置 | 结果 | 能说的话 |
| --- | --- | --- | --- |
| 可行性 | AQUA-RAT test 随机 100，贪心，最后 30% token 当 Phase III | 0.5B 第 16 层 \(\Delta\mathrm{NDI}=+0.038\)，\(d=1.45\)；1.5B 第 18 层 \(\Delta=+0.021\)，\(d=1.09\) | 比例切分下，结尾相对中间段离开任务向量。两个尺度方向一致。 |
| Stage A | 同一 100 条，留出 8 条造任务向量，评 92 条；内容切分 | 0.5B：\(\Delta=0.024\)，\(d=1.59\)，\(p=5\times10^{-27}\)。1.5B：\(\Delta=0.015\)，\(d=0.60\)。随机向量不支持 \(H_1\)。乱序选项仍支持 \(H_1\)（0.5B \(d=1.51\)），`control_artifact=true` | 方向不是任意向量的伪影。它也没有被「打乱选项正文」消掉。行为上 0.5B 字母准确率 17.4%，错配率 8.7%，数值命中 15.2%，格式合法率 6.5%。 |
| Stage B | 0.5B LoRA，AQUA train 100 构造偏好，test 100 | Base / SFT / DPO / Rep-DPO 的错配率 0.10 / 0.11 / 0.15 / **0.30**。Rep-DPO 的 Phase III NDI 升到 0.070，\(\Delta\) 反转为 \(-0.011\)，数值命中 0.15→0.38，准确率 0.16→0.24 | 表征项把中间数值和 Phase III 投影拉上去了。字母对齐更差。预先写的主成功条件（错配下降）未达到。 |

协议里已经有的部件：任务内严谨向量、内容切分、字母 / 数值 / 错配解析、随机向量与置换向量、选项正文打乱、Phase III mask 的表征 hinge、SFT / 配对排序 / Rep-DPO。

协议里写了但没有进冻结表的部件：无选项续写、MathQA / GSM8K-MCQ / 常识 MCQ、1.5B 修复、Rep-GRPO、三种子、人工切分 \(\kappa\)、层扫描主表（代码扫了层，合并报告只留了主层）。

## 2. 对照物：一篇靠得住的同题论文

质量标尺用 Li 等人的 *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*（NeurIPS 2023 spotlight）。它和这篇工作是同一类论证：对比激活找出一个方向，再干预这个方向，让表面行为和内部方向一起变。NeurIPS 在 CCF 目录里是 A 类。

ITI 做成、而当前冻结结果还没有的部分：

1. 标题指标是行为，不是探针。Alpaca 上 TruthfulQA 真实率从 32.5% 到 65.1%。探针只用来找方向。
2. 干预是因果的。推理时沿该方向平移激活，行为随之变；强度可扫，并单独报告有用性代价。
3. 方向在留出题上选定，再在完整基准上评。TruthfulQA 有 817 题、生成与选择两条轨道。
4. 模型尺度到指令微调的 LLaMA，而不是只停在 0.5B。
5. 随机方向和未干预模型都在同一张表里。

另一篇必须同时挡住的混淆是 Zheng 等人 *Can Multiple-choice Questions Really Be Useful in Detecting the Abilities of LLMs?*（LREC-COLING 2024）。他们证明模型对选项顺序有稳定偏好，而且打乱的是选项内容而不只是字母。当前 Stage A 的乱序对照已经朝这个方向做了，但结果是漂移几乎不变。按实验设计自己的失败解释，这更像「收尾动作 / 结束位置」效应，而不是「把算对的数对错了字母」。

## 3. 相对这篇标尺，现在还缺什么

下面每一条都对应仓库里的具体实现或冻结数字，不是写作口味。

1. **主结论被自己的对照否定了。** `reports/stage_a/stage_a_h1.json` 里两个模型都是 `h1_supported=true` 且 `control_artifact=true`。乱序后 0.5B 的 \(d\) 从 1.59 只降到 1.51。设计文档第 3 节写明：乱序选项若仍有同样漂移，就不能把现象叫作选项匹配。在补上「无选项续写」和「与答案 token 等长的非选项结尾」之前，论文不能把 \(\Delta\mathrm{NDI}\) 写成 option mismatch。

2. **要修的行为很少，修完之后更差。** 0.5B 基线错配只有 8.7%（92 条里大约 8 条），主导错误是「根本没算对 / 没按格式作答」（准确率 17%，格式合法率 6.5%）。Stage B 的 Rep-DPO 把错配从 10% 抬到 30%，数值命中从 15% 抬到 38%。这符合设计文档预先登记的失败：表征变好、字母对齐没跟上。现在的贡献是「更会吐中间数」，不是「算对了却选错被修好了」。

3. **偏好对把两件不同的事捆在一起。** `preferences.py` 的 chosen 是金标准解答全文加 `Final answer`，rejected 是模型自己的生成。模型要同时模仿金标准推理和收尾。设计文档第 5.5 节禁止的正是这种全文替换。字母一致性没有进损失。\(\tau\) 仍写死为 0.054（`configs/qwen25_0.5b_2080.yaml`），不是该模型 Phase II 均值。

4. **名叫 DPO 的损失不是 DPO。** `train_rep_dpo.py` 的配对项是 \(\log\sigma(\beta(\log\pi(y_w)-\log\pi(y_l)))\)，没有冻结的参考模型，因此没有 \(\log\pi_{\mathrm{ref}}\)。这是策略自身对数概率的排序，外加一个 Phase III hinge。和 Rafailov 等人的 DPO 不能放在同一行基线里。审稿人会先卡这一点。

5. **z-score 表不提供新信息。** `_layer_report` 把每条样本的 Phase II 均值和 Phase III 均值拼成一个向量再标准化。两组样本数相同，所以标准化后的两组均值必然互为相反数。Stage A 里 0.5B 的 z-scored 均值 \(+0.582\) 与 \(-0.582\) 就是这个代数结果。主表只应报原始 \(\Delta\)、\(d\) 和层内比较，不能把这列写成独立的尺度校正。

6. **1.5B 上「严谨方向」在 Phase II 已经是负的。** Stage A 的 1.5B Phase II 均值是 \(-0.014\)，Phase III 是 \(-0.029\)。\(\Delta>0\) 只说明结尾更负，不能说 Phase II「保持高 NDI」。绝对投影不能跨模型比较，这点设计文档已经写了，正文若仍把 Phase II 均值解释成严谨程度就会过界。

7. **证据规模停在可行性。** 一个数据集、一个种子、\(N=92/100\)、修复只在 0.5B。没有迁移集，没有无选项对照，没有切分一致性，没有三种子置信区间。层扫描没有进入合并后的主 JSON。

8. **NLPCC 本身不是目标档。** 本地稿若按 NLPCC 写，NLPCC 在 CCF 目录里是 C 类。下面第 4 节的门槛按 CCF-A 会议和 CCF-B / 中科院 1 区期刊设，不按 NLPCC 页数设。

## 4. 锁定的投稿目标

分区口径：中科院分区表自 2026 年起停更，学校目前通行的是 2025 表。CCF 以 2026-03-31 发布的第七版为准。第七版公开说明里，ICLR 由 B 升为 A，IJCAI 由 A 降为 B。NAACL、COLING 在 2026-08 的 CCF 截稿清单中仍为 B。Findings、workshop、short paper 不计入 CCF 目录。

已经关掉、不要再追的窗口：

| 出口 | 级别 | 为什么现在不追 |
| --- | --- | --- |
| ICLR 2027 | CCF-A | 全文截稿 2026-09-25 AOE，已过 |
| AAAI 2027 | CCF-A | 社区截稿表记为 2026-07-29，已过 |
| EACL 2027 | CCF-B | ARR 最后投稿日 2026-08-03，承诺日 2026-10-11。新实验来不及进这一轮 |
| NAACL 2027、COLING 2027 | CCF-B | ARR 投稿日 **2026-10-12**，承诺日 2026-12-23（[ARR dates](http://aclrollingreview.org/dates)）。距今约两周。Stage B 主指标是负的，这一轮投出去会被自己的表拒掉 |

锁定两个出口，共用同一套实验门槛。先满足门槛，再按出口写。

**主目标：ACL 2027（CCF-A）。** 会议在京都，2027-08-17 至 2027-08-22。审稿走 ARR，官方表把 ACL 2027 的最后 ARR 投稿月写成 2027 年 1 月，具体日期尚未公布，承诺日仍空着。这是主题匹配、级别达标、而且尚未关闭的会议。1 月投稿意味着 2026 年 12 月前主表必须冻结。

**并行目标：IEEE Transactions on Neural Networks and Learning Systems（TNNLS）。** 公开目录把它标为 CCF-B（人工智能）、JCR Q1、2025 中科院计算机科学 1 区 TOP。没有固定截稿日，和 ACL 用同一套实验。若 1 月 ARR 的行为结果仍不干净，就走 TNNLS，而不是降到 C 类会议。投稿前用学校图书馆打开的 2025 分区表再核一次 TNNLS 的大类分区。

不把 TASLP 列为主目标：它同样常被标成 CCF-B、JCR Q1、中科院 2 区，但版面偏语音与语言处理系统，和「隐状态加偏好损失」的契合度不如 TNNLS。

## 5. 实验目标

下面的判定在跑之前写死。任何一条达不到，就不把该设置写进 ACL / TNNLS 主表。

### E0. 先做因果转向，再决定要不要继续训

在冻结的 0.5B 与 1.5B 上，只在 Phase III 的隐状态上加上 \(\alpha V_\ell\)，\(\alpha\) 取一小段正负扫描。对照是同一层的随机单位向量。

出门条件：在「Phase II 数值已经命中某个选项」的子集上，正 \(\alpha\) 使错配率下降，随机向量不下降。若转向不动字母，Rep-DPO 沿这个方向加 hinge 就没有因果基础，后面的训练目标作废，论文改口为结尾位置效应，并停止把贡献叫作 option mismatch。

### E1. 把现象从「结尾效应」里拆出来

补两类对照，仍用内容切分，报告主层和全部扫描层：

- 去掉选项，只要求输出数值。
- 保留选项，但把 Phase III 切点对齐到一个与答案句等长、不含 A–E 的收尾（例如复述最后一步算术）。

出门条件：任务向量上的 \(\Delta\mathrm{NDI}\) 在这两种对照里相对主条件明显变小（Cohen's \(d\) 降到主条件的一半以下，或 \(H_1\) 不再成立）；随机向量上 \(H_1\) 继续不成立。做不到这一点，E0 即使成功，也只能声称「收尾段可转向」，不能声称「选项映射段塌缩」。

### E2. 损失和偏好要和主张一致

- 参考模型冻结，配对项改成标准 DPO：\(\log\sigma\big(\beta[(\log\pi-\log\pi_{\mathrm{ref}})_w-(\log\pi-\log\pi_{\mathrm{ref}})_l]\big)\)。
- \(\lambda_{\mathrm{rep}}\in\{0, 0.5, 1.0\}\)，\(0\) 就是这条实现下的普通 DPO。
- \(\tau\) 用该模型、该层在训练集上的 Phase II 均值，不写死 0.054。
- chosen 与 rejected 共享同一段 Phase II 文本。差异只发生在 Phase III：chosen 把已命中的数值对到正确字母，rejected 是模型真实的错配收尾。禁止用整段金标准解答替换推理。
- 表征 hinge 只反传到 Phase III token。另加一项字母一致性：最终字母与 Phase II 所命中的选项不一致则惩罚。

### E3. 行为主指标，只在「本来会做」的子集上判定

预先定义可修复子集：Phase II 抽出的数值能对上某个选项。在这个子集上报告错配率。全集上报告字母准确率、数值命中、格式合法率，避免只在容易题上说话。

出门条件，三种子 \(\{42,43,44\}\) 的均值，并报 95% 置信区间：

- 可修复子集上，Rep-DPO 的错配率比 Base 和比 \(\lambda=0\) 的 DPO 都至少低 10 个百分点。
- 全集字母准确率不低于 Base。
- Phase III NDI 高于 Base，\(\Delta\mathrm{NDI}\) 小于 Base。
- SFT 与 Phase-III-only SFT 都在表里。若只有 SFT 达标，贡献不成立。

Stage B 的现状（错配 +20 个百分点）不满足这条。在 E2 落地并重跑之前，不把 Stage B 的 Rep-DPO 数字写进投稿主表。

### E4. 规模

| 项 | ACL 2027 主表 | 若只够 TNNLS，仍不得低于 |
| --- | --- | --- |
| 修复模型 | Qwen2.5-1.5B 全套消融；7B 做 LoRA 修复或至少做 E0 转向。V100 32GB 上 7B LoRA 用梯度检查点，不再把 7B 写成不可做 | 1.5B 全套消融，另加 3B 的 E0 与评测 |
| 主评测 | AQUA-RAT 全部 test（254），不是 100 | 同左 |
| 训练 | 去重后的 AQUA train，至少 800 条偏好，且只保留「Phase II 命中、收尾可分对错」的对 | 至少 400 条这种对 |
| 迁移 | MathQA test 400。训练题与测试题去重 | MathQA 200 |
| 域外 | ARC-Easy 或 CommonsenseQA 400，只做探针和行为，不要求迁移成功；失败要写成边界 | 同左，可降到 200 |
| 切分 | 50 条人工核对，\(\kappa\ge 0.8\)。回退到最后 30% 的比例单独报告，不进主结论 | 同左 |
| 统计 | 层扫描用 BH-FDR。主文报配对检验、\(d\) 和三种子区间。不报第 3 节指出的那种 z-score 均值 | 同左 |

### E5. 写作冻结

主表只冻结一次，对应设计文档的 Stage D。E0 或 E1 失败时，改的是主张，不是阈值。补充实验不得再改 Phase III 的主切分规则。

图只保留四张，和主张一一对应：单条 NDI 轨迹并标出 Phase III 起点；层间 \(\Delta\)；转向或 Rep-DPO 前后的配对点；可修复子集上错配与 \(\Delta\mathrm{NDI}\) 的散点。散点相关若接近 0，正文不能写「塌缩导致错配」。

## 6. 执行顺序

1. E0 转向扫描。这是最便宜的因果检验，决定后面还是不是这篇论文。
2. E1 无选项与等长收尾。与 E0 共用同一批生成缓存。
3. 按 E2 改训练代码后，先在 0.5B、AQUA 100 上重跑 E3 的方向。错配不再上升，才扩到 1.5B 和 E4 的 \(N\)。
4. 2026 年 12 月冻结主表。1 月 ARR 投 ACL 2027。同一包结果同时整理 TNNLS 版本，避免 1 月窗口错过之后从零再做实验。
