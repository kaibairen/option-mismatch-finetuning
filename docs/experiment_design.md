# 选项错配 / 严谨度漂移：完整实验分析与技术方案

状态：V100 可行性已通过（AQUA-RAT 随机 100 条；0.5B / 1.5B 均支持 $H_1$）。  
本文把「Phase III 粗心收尾」定为正式切入点，并给出可在 Tesla V100-32GB 上闭环的实验与实现方案。

## 1. 结论：这是不是切入点

**是。** 而且应当把切入点收窄成一句话：

> 模型在单步 CoT 演绎（Phase II）仍保持可分的严谨表征，但在把中间结果对齐到 A/B/C/D/E（Phase III）时发生可统计的表征塌缩；该塌缩是**局部激活漂移**，不是知识缺失，因此可用针对 Phase III 隐状态的后训练修复。

支持这一收窄的证据：

| 来源 | 结果 | 含义 |
| --- | --- | --- |
| 原报告（0.5B × AQUA 30） | $\Delta\mathrm{NDI}=+0.0091$，$p=0.0033$，$d=0.53$ | 现象首次被宣称 |
| 本仓库 V100 复现（0.5B × 100） | $\Delta=+0.0376$，$p=1.4\times10^{-26}$，$d=1.45$ | 同模型、同任务上方向一致且更强 |
| 本仓库 V100 复现（1.5B × 100） | $\Delta=+0.0205$，$p=6.8\times10^{-19}$，$d=1.09$ | 不是 0.5B 特例 |

**为什么这是好切入点（而不是「再训一遍数学」）：**

1. **可定位**：错误集中在选项匹配段，而不是整段推理崩溃。
2. **可测量**：同一条轨迹上 Phase II / III 成对比较，统计功效高。
3. **可干预**：目标是一层或数层的隐状态，而不是重写世界知识。
4. **和仓库目标对齐**：`option-mismatch-finetuning` 的产品假设就是「算对了却选错」。

**现在还不能当论文终局的缺口（必须在正式实验里补上）：**

- 绝对 NDI 与原报告不可直接对比（探针构造不同）。
- Phase III 目前用「最后 30% token」切分，偏粗糙。
- 还缺**行为指标**：选项正确率、选项错配率（推理数字对但字母错）。
- 还缺对照：随机投影、乱序选项、无选项生成、SFT-only、普通 DPO。
- 1.5B 的 Phase II 均值接近 0，说明向量尺度会随模型变，必须层内标准化 + 多层报告。

正式实验必须同时报 **表征指标（NDI）** 和 **行为指标（Accuracy / Mismatch）**。只报 $p$ 值不够。

---

## 2. 问题形式化

记模型在第 $\ell$ 层、第 $t$ 个生成 token 的隐状态为 $h_\ell^{(t)}\in\mathbb{R}^{d}$。  
严谨向量 $V_\ell$ 由对照提示的末 token 隐状态差分得到：

$$
V_\ell=\mathrm{normalize}\Big(\mathbb{E}[h_\ell^{\mathrm{diligent}}]-\mathbb{E}[h_\ell^{\mathrm{careless}}]\Big)
$$

$$
\mathrm{NDI}(\ell,t)=\langle \mathrm{normalize}(h_\ell^{(t)}),\,V_\ell\rangle
$$

一条解答轨迹按内容切成三段（实现上先规则、再人工抽检）：

| 阶段 | 内容 | 科学含义 |
| --- | --- | --- |
| Phase I | 复述题面 / 设未知数 | 读题，不作为主假设 |
| Phase II | 逐步演算到中间数值 | 演绎，应保持高 NDI |
| Phase III | 「所以选 X」/ 把数值对到选项 | 收尾匹配，假设发生塌缩 |

主效应：

$$
\Delta\mathrm{NDI}_i=\overline{\mathrm{NDI}}_{\mathrm{II},i}-\overline{\mathrm{NDI}}_{\mathrm{III},i}
$$

**选项错配（Option Mismatch）** 是行为侧主指标，定义比「答错」更严：

- 轨迹中抽出的中间数值 / 结论与某选项一致，但最终字母不是该选项；或
- 最终字母与 gold 不符，且 Phase II 的中间结论本可映射到 gold。

这把「不会做」和「会做但收尾选错」拆开。后者才是本项目要修的。

---

## 3. 假设体系

| ID | 陈述 | 判定 |
| --- | --- | --- |
| $H_1$ | 固定模型与 MCQ 轨迹上，$\mathbb{E}[\Delta\mathrm{NDI}]>0$（Phase III 相对 Phase II 下降），配对 $t$ 与 Wilcoxon 均 $p<0.005$，且 $d>0.3$ | **可行性已支持**；正式实验需内容切分 + 多层 + 多数据集复验 |
| $H_2$ | 针对 Phase III 的 Rep-DPO / Rep-GRPO 使 $\Delta\mathrm{NDI}$ 下降，同时 **Mismatch↓、Accuracy↑**，且不低于普通 DPO / SFT | 待做（核心贡献） |
| $H_3$ | 修复可迁移：AQUA 上得到的 LoRA，在 MathQA / GSM8K-MCQ 上仍降低 Mismatch，且通用指令准确率不明显掉 | 待做（证明不是背数据集） |
| $H_0^{\mathrm{ctrl}}$ | 随机向量、乱序选项、无选项续写上，$\Delta\mathrm{NDI}$ 消失或显著变弱 | 必须做；用来排除「凡是结尾都掉」的假象 |

预先登记的失败解释：

- 若只有 NDI 变好、Accuracy 不变：表征对齐是装饰，方案不成立。
- 若 Accuracy 升但 Mismatch 不变：只是更会猜字母。
- 若乱序选项仍有同样漂移：可能是「结束符/长度」伪影，不是选项匹配。

---

## 4. 实验矩阵

### 4.1 模型（V100-32GB 约束）

| 阶段 | 模型 | 权重约 | 角色 |
| --- | --- | --- | --- |
| 已完成可行性 | Qwen2.5-0.5B-Instruct | 1 GB | 主复现、最快迭代 |
| 已完成可行性 | Qwen2.5-1.5B-Instruct | 3 GB | 同家族尺度 sanity |
| 正式主实验 | 继续 0.5B + 1.5B | — | 全协议、消融、修复 |
| 可选加固 | Qwen2.5-3B-Instruct | ~6 GB | 只做 probe + 评估，不做重训 |

7B 全参不在本卡预算内；3B 仅推理 + LoRA 评估。

### 4.2 数据

每个数据集先固定 **seed=42 的 100 条** 跑通协议，再扩到正式 $N$。

| 数据集 | 类型 | 抽样 | 正式 $N$ | 用途 |
| --- | --- | --- | --- | --- |
| AQUA-RAT test | 5 选 1 数学 | 100（已有） | 全 test 254 + 分层 400 train | 主 benchmark |
| AQUA-RAT train | 同上 | 100（已有） | 800–2000 | 后训练 |
| MathQA | 5 选 1 | 100 | 400 test / 800 train | 近迁移 |
| ARC-Easy 或 CSQA | 常识 MCQ | 100 | 400 | 跨域：漂移是否不止数学 |
| GSM8K（改写成 4 选项） | 开放改 MCQ | 100 | 400 | 控制「选项密度」 |

分层：按题干长度与选项数字跨度分层，避免 100 条全是短题。

### 4.3 自变量

- 模型尺度：0.5B / 1.5B（/ 3B）
- 阶段定义：token 比例切 vs 内容切
- 层：0.5B 报 8/12/16/20/24；1.5B 报 10/14/18/22/28
- 后训练：无 / SFT / DPO / Rep-DPO / Rep-GRPO
- 对照：随机 $V_\ell$、乱序选项、去掉选项只出数字

### 4.4 因变量

**表征**

- $\overline{\mathrm{NDI}}_{\mathrm{II}}$、$\overline{\mathrm{NDI}}_{\mathrm{III}}$、$\Delta\mathrm{NDI}$
- 层间可分度（线性探针或质心间隔）
- Phase III 段 NDI 斜率（是否在第一次出现 A/B/C/D/E 后急降）

**行为**

- Letter Accuracy
- Option Mismatch Rate
- Exact-number match（抽出的中间数值是否等于 gold 选项数值）
- 格式合法率（是否输出 `Final answer: X`）

**统计**

- 配对 $t$（单侧 greater）、Wilcoxon 符号秩
- Cohen's $d_z$
- 预注册阈值：$p<0.005$ 且 $d>0.3$ 才称「支持 $H_1$」
- 多数据集 / 多层用 BH-FDR
- 修复实验报 95% CI，种子 $\{42,43,44\}$

---

## 5. 技术方案

### 5.1 总流程

```text
数据集 → 分层抽样 → 贪心 CoT 生成
        → 内容切分 Phase II/III
        → 抽 h_ℓ 并投到 V_ℓ 得 NDI 轨迹
        → 行为标注（字母 / 数值 / 错配）
        → H1 统计 + 对照
        → 构造 preference（chosen=严谨收尾, rejected=塌缩收尾）
        → LoRA Rep-DPO / Rep-GRPO（只加强 Phase III）
        → 同探针复测 NDI + Accuracy + Mismatch
        → 迁移集与通用集回归
```

### 5.2 严谨向量（必须比可行性更干净）

可行性用了 8 对合成中英对照句。正式实验改为：

1. **任务内对照**：同一道 AQUA 题，系统提示分别为 diligent / careless，取答案开始前最后一个 prompt token 的 $h_\ell$。
2. **留一**：构造 $V_\ell$ 的题不得进入 probe 统计。
3. **白化**：对同层 probe 集做均值中心化后再单位化，消除 1.5B 上「均值贴 0」的尺度问题。
4. **负对照向量**：高斯随机方向、$V_\ell$ 分量重排；期望 $\Delta\mathrm{NDI}\approx 0$。

### 5.3 Phase 切分（从比例改为内容）

主切分规则（按优先级）：

1. 第一次出现 `Final answer` / `答案是` / `所以选` / 独立字母行的 token 起为 Phase III。
2. 否则：第一次把中间结果显式对到选项编号的 span。
3. 再否则：回退最后 30%（只当灵敏度分析，不当主结论）。

人工抽 50 条核切分 κ，目标 $\kappa\ge 0.8$。

### 5.4 生成协议

- 解码：贪心（与可行性一致），`max_new_tokens=256`（正式；可行性 128 只作速度折中）
- 温度采样只用于构造 DPO rejected，不用于主 $H_1$ 表
- 固定 chat template、seed、停止符
- 缓存 `results/generations/{model}/{dataset}/{split}.jsonl`，表征与训练复用同一批文本

### 5.5 后训练：Rep-DPO / Rep-GRPO

在现有 `train_rep_dpo.py` 上收紧，不要再只用「我随便选个字母」这种假 rejected。

**Preference 构造**

- chosen：gold rationale + `Final answer: {gold}`；或模型自己 Phase II 正确且收尾正确的轨迹
- rejected：同 prompt 下模型真实塌缩轨迹（Mismatch=1 或 Phase III NDI 最低的一条）
- 禁止：与 chosen 只差一个字母、但推理全文复制——那会变成字母分类器

**损失**

$$
\mathcal{L}=\mathcal{L}_{\mathrm{DPO}}+\lambda_{\mathrm{rep}}\cdot\mathbb{E}\big[\mathrm{ReLU}(\tau-\overline{\mathrm{NDI}}_{\mathrm{III}}^{\mathrm{chosen}})\big]
+\lambda_{\mathrm{kl}}\cdot\mathrm{KL}(\pi\Vert\pi_{\mathrm{ref}})
$$

- $\tau$：该模型 Phase II 均值（0.5B 可行性约 0.054；不要再写死原报告 0.1126）
- $\lambda_{\mathrm{rep}}\in\{0.0,0.5,1.0\}$ 扫描；$0$ 即普通 DPO 基线
- LoRA：$r=16$，$\alpha=32$，目标模块 q/k/v/o/gate/up/down
- 只对 Phase III token 反传表征项（mask 掉 Phase II），这是和「全序列 SFT」的关键差别

**Rep-GRPO（第二阶段）**

奖励：

$$
R=1_{\mathrm{correct}}+\alpha\cdot\overline{\mathrm{NDI}}_{\mathrm{III}}-\beta\cdot 1_{\mathrm{mismatch}}
$$

组内 4 条采样，V100 上只对 0.5B 做；1.5B 以 Rep-DPO 为主。

### 5.6 基线

| 方法 | 目的 |
| --- | --- |
| Base Instruct | 起点 |
| SFT on gold rationale | 排除「多看点解答就行」 |
| 普通 DPO（$\lambda_{\mathrm{rep}}=0$） | 排除「只要 preference 就行」 |
| Phase-III-only SFT | 排除「只练结尾格式」 |
| Rep-DPO | 主方法 |
| Rep-GRPO | 0.5B 加强 |

### 5.7 V100 工程约束

| 资源 | 用法 |
| --- | --- |
| GPU 32 GB | 1.5B LoRA + 隐状态完全够；3B 只评估 |
| `/` 30 GB | 不放权重 |
| `/root/autodl-tmp` 50 GB | 模型、HF/MS 缓存、结果 |
| 网络 | 只走 ModelScope，不走 Hugging Face |
| 运行目录 | `/root/autodl-tmp/option-mismatch-finetuning` |

目录约定：

```text
/root/autodl-tmp/
  models/Qwen2.5-*-Instruct/
  option-mismatch-finetuning/
    data/processed/{aqua,mathqa}_train_100.jsonl
    results/generations/
    results/probes/
    results/adapters/rep_dpo_0.5b/
    reports/
```

---

## 6. 分阶段执行（按依赖，不按日历）

### Stage A — 协议硬化（当前之后立刻做）

1. 内容切分 Phase III + 50 条人工核验
2. 行为解析器：字母、数值、Mismatch
3. 层扫描 + 随机向量对照
4. 乱序选项 / 无选项续写
5. 把 0.5B / 1.5B 的 100 条按新协议重打分（同一批 generation 可复用）

出门条件：新协议下 $H_1$ 仍成立；乱序或随机向量上 $\Delta$ 明显变弱。

### Stage B — 100 条修复可行性

1. 用 100 条 AQUA train 构造真实 preference
2. 0.5B 上 LoRA Rep-DPO（约 1 epoch）
3. 在 AQUA test-100 上复测 NDI + Accuracy + Mismatch
4. 对照普通 DPO、SFT

出门条件：Mismatch 或 Accuracy 至少一项优于 DPO，且 Phase III NDI 回升。

### Stage C — 正式规模

1. AQUA 扩大 $N$；加入 MathQA 100→400
2. 1.5B Rep-DPO
3. 三种子、FDR、CI
4. 迁移 + 通用回归（抽 100 条普通指令，防崩）

### Stage D — 写作冻结

主表只冻结一次。补充实验不得再改主切分规则。

---

## 7. 主结果表（预留）

**表 1 — $H_1$ 表征**

模型 × 数据集 × 切分规则：$\mathrm{NDI}_{II}$、$\mathrm{NDI}_{III}$、$\Delta$、$t$、$p$、$d$。

**表 2 — 行为**

Accuracy、Mismatch、Number-match、Format。

**表 3 — 修复消融**

Base / SFT / DPO / Rep-DPO / Rep-GRPO：$\Delta\mathrm{NDI}$、Mismatch、Accuracy。

**图**

1. 单条轨迹 NDI(t)，竖线标 Phase III 起点  
2. 层间 $\Delta\mathrm{NDI}$ 热力  
3. 修复前后 Phase III 分布（配对点）  
4. Mismatch vs $\Delta\mathrm{NDI}$ 散点（应正相关）

图 4 是机制图：若相关为 0，$H_2$ 的因果叙事不成立。

---

## 8. 效度威胁与处理

| 威胁 | 处理 |
| --- | --- |
| 结尾 token 天然更「散」 | 乱序选项、无选项、随机 $V$ |
| 合成对照句泄漏风格 | 任务内对照 + 留一 |
| 切分规则偏向 | 双规则报告，主结论用内容切 |
| 小模型本来就不会做 | 同时报 Number-match；只在「会算」子集上看 Mismatch |
| NDI 尺度不可比 | 层内 z-score；只比较 $\Delta$ 与 $d$ |
| 后训练背答案 | 迁移集 + 训练题与测试题去重 |
| 解码噪声 | $H_1$ 主表贪心；采样只用于 RL |

---

## 9. 与可行性代码的差距（实现清单）

现有代码已具备：向量差分、token NDI、配对检验、LoRA DPO+表征 hinge、V100 下载脚本。

必须补的（按优先级）：

1. `span.py`：内容切 Phase III，而不是 `phase3_token_frac=0.30`
2. `behavior.py`：字母 / 数值 / Mismatch 解析
3. `controls.py`：随机向量、乱序选项
4. 真实 rejected 采样，替换 `train_rep_dpo.py` 里的假收尾
5. 表征损失只作用于 Phase III mask
6. $\tau$ 改为该模型 Phase II 均值
7. MathQA 抽样器
8. 层扫描与轨迹作图

---

## 10. 论文叙事（建议只讲一个故事）

标题方向：*Option Mismatch as Phase-III Diligence Collapse, and Repairing It with Representation-Aware DPO*。

1. MCQ 上存在「算到了却选错」  
2. 该错误对应 Phase III 隐状态离开严谨方向  
3. 不是缺知识，因为 Phase II 仍可分  
4. 对 Phase III 加表征约束的 DPO 能同时拉回 NDI 并降低 Mismatch  
5. 可迁移，且随机/乱序对照排除伪影  

不把 NDI 写成「情绪」。对外表述用 **diligence / hasty-commitment representation**，对内可沿用「严谨 / 急躁」。

---

## 11. 当下决策

1. **切入点锁定**：Phase III 选项匹配段的表征塌缩 + 选项错配。  
2. **不再扩大模型** 直到 Stage A 对照做完。  
3. **下一步实现**：内容切分、行为指标、对照实验，然后用已有 100 条 train 做 0.5B Rep-DPO 可行性。
