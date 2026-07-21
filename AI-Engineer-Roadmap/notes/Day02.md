# Day 2：Prompt Engineering 系统方法论

**日期**：2026-07-09

---

## 1. 核心概念

### 1.1 什么是 Prompt Engineering？

大多数人的 Prompt 是错的：

```
❌ 差 Prompt：
写一篇关于机器学习的文章。

✅ 好 Prompt：
## Role
资深技术博主
## Task
写一篇面向数据分析师的机器学习入门文章
## Constraints
- 1000 字以内
- 用类比解释复杂概念
- 给出一个实际 Python 代码示例
```

**Prompt Engineering = 设计和优化给 LLM 指令的系统方法**。它不是"和 AI 聊天"，而是用结构化的方式给模型"编程"。

### 1.2 Prompt 的 5 大核心技术

| 技术 | 核心思想 | 适用场景 |
|------|---------|---------|
| **Persona** | 给模型一个角色（你是谁？） | 所有场景的基础 |
| **Few-shot** | 给示例让模型模仿模式 | 分类、抽取、格式化 |
| **Chain-of-Thought** | 让模型逐步推理 | 数学、逻辑、风控分析 |
| **Structured Prompt** | 用 Markdown/XML 组织 | 复杂任务、多步骤 |
| **Self-Consistency** | 多次采样取多数 | 需要高准确率 |

### 1.3 Chain-of-Thought（思维链）深入分析

**原理**：让模型"展示推理过程"而不是直接给答案。

```
❌ 不推荐（直接输出）：
Q: 一个水池进水 3h 满，排水 5h 空，同时开多久满？
A: 7.5h

✅ 推荐（逐步推理）：
Q: 同上
A: 进水速度 = 1/3 池/小时
   排水速度 = 1/5 池/小时
   净速度 = 1/3 - 1/5 = 2/15 池/小时
   时间 = 1 / (2/15) = 7.5 小时
   所以答案是 7.5h
```

**为什么 CoT 有效？** 三个原因：

1. **延长计算深度**：每一步推理都是一次"前向传播"，更多步骤 = 更充分的内部计算
2. **分解复杂度**：把"判断风险"分解为"时间→设备→金额→收款方"
3. **可校验性**：中间步骤可以人工审查，发现推理错误

**失效场景：**
- 问题本身不需要推理（如"北京的邮编是多少"）
- 模型在中间步骤已经错了，错误会累积
- 需要外部知识或事实性查证

### 1.4 Structured Prompt 模式（工程中最重要）

这是工程中实际使用的 Prompt 写法——可维护、可复用、可测试：

```markdown
## Role
[角色定义]

## Task
[任务描述]

## Context
[背景信息]

## Constraints
- 要求 1
- 要求 2

## Output Format
```json
{
  "field1": "..."
}
```

## Examples
输入: ...
输出: ...
```

**和代码的类比：**

| 代码概念 | Prompt 对应 |
|---------|------------|
| 函数签名 | Role + Task |
| 参数校验 | Constraints |
| 返回类型 | Output Format |
| 单元测试 | Examples |
| 文档注释 | Context |

### 1.5 Few-shot 的核心洞察

**示例比描述更有效。** 模型从示例中学习格式和模式，比从自然语言描述中学习更准确。

```python
# Description 方式（不够好）：
"输出 JSON 格式，包含 person_name, date, amount, risk_type 字段"

# Few-shot 方式（更准确）：
"""
输入: 客户张伟于2024年3月15日投诉账户被盗
输出: {"person_name": "张伟", "date": "2024-03-15", "amount": null, "risk_type": "盗刷"}
输入: 今天收到一条短信
输出: {"person_name": null, "date": null, "amount": null, "risk_type": "钓鱼"}
"""
```

示例相当于"类型声明 + 默认值"——模型直接看到你要什么格式，而不是"听你描述"。

### 1.6 Self-Consistency

Self-Consistency = 多次 CoT 推理 + 取多数答案。CoT 是单路径推理，Self-Consistency 是多路径投票。

```
问题: 逻辑推理题

CoT 路径 1: 推理步骤为 A→B→C→D→结果 X
CoT 路径 2: 推理步骤为 A→B'→C'→D'→结果 Y
CoT 路径 3: 推理步骤为 A→B→C→D→结果 X  ← 多数

最终答案: X（取多数）
```

优点是降低随机错误，缺点是成本是 N 倍 API 调用。

---

## 2. Coding 实战

### 2.1 PromptTemplate —— 把 Prompt 当作代码组件

**文件**：`demos/day02/prompt_builder.py`

```python
@dataclass
class PromptTemplate:
    """结构化 Prompt 模板"""
    role: str = ""
    task: str = ""
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    output_format: str = ""
    examples: list[tuple[str, str]] = field(default_factory=list)

    def build(self, **kwargs) -> str:
        """组装最终 Prompt，支持动态变量填充 {var_name}"""
        parts = []
        if self.role:
            parts.append(f"## Role\n{self.role}")
        if self.task:
            # 支持 {variable} 动态替换
            task = self.task.format(**kwargs) if kwargs else self.task
            parts.append(f"## Task\n{task}")
        # ... 依次组装各部分
        return "\n\n".join(parts)
```

**设计思想**：
- 组件化：每个 Prompt 部分可独立修改、测试
- 可复用：预置模板（risk_analysis / chain_of_thought / extractor）
- 可组合：build() 最后组装成完整 Prompt

**预置的三种模板：**

```python
def risk_analysis_template() -> PromptTemplate:
    """风控分析模板：Persona + Task + Constraints + OutputFormat"""

def chain_of_thought_template() -> PromptTemplate:
    """思维链模板：要求逐步推理"""

def extractor_template() -> PromptTemplate:
    """信息抽取模板：带 Few-shot 示例"""
```

### 2.2 对比实验 —— 5 种策略实测

**文件**：`demos/day02/experiment.py`

测试数据：用户王强在凌晨 2:30 使用新设备登录，转账 50000 元到新账户

| 策略 | 回复长度 | 特点 |
|------|---------|------|
| ① 简单问答（基线） | 114字 | 笼统，无结构，直接给结论 |
| ② 结构化 Prompt | 514字 | 有 Risk Level / 理由 / 建议 / 缺失项 |
| ③ CoT 思维链 | 641字 | 逐步推理，最详细，可追溯 |
| ④ Few-shot 抽取 | 82字 | 精确 JSON，适合机器消费 |
| ⑤ 多角度评估 | 461字 | 分 4 维度打分（19/20），定量 |

**关键观察：**

1. **Strategy 2 vs Strategy 1 的差异**：结构化 Prompt 的回复里包含了"缺少信息"部分，而简单问答没有提到任何缺失信息。原因就是 Constraints 中写了"如果信息不足，明确指出缺少什么信息"——模型严格遵循了这行"代码"约束。

2. **Strategy 3 (CoT) 的推理过程**：
   ```
   Step 1: 分析时间因素 → 凌晨 2:30 属于高风险时段
   Step 2: 分析设备风险 → 新设备登录无历史记录
   Step 3: 分析收款方 → 30 分钟前新注册
   Step 4: 综合分析 → 多重异常叠加，结论为"高风险"
   ```
   每一步都有明确依据，且可追溯到具体输入特征。

3. **Strategy 4 (Few-shot) 输出 JSON**：直接输出结构化数据，适合后续程序化处理。

---

## 3. Prompt 为什么"有用"？—— 源码分析

### 3.1 本质

Prompt 不是"和 AI 聊天"，它是**给模型的隐式编程指令**。

```
"分析这个交易"
    → 模型内部：这是什么任务？什么格式？多详细？→ 自己猜（可能猜错）

结构化 Prompt
    → 模型内部：Persona=风控专家，Task=多角度分析，
      Format=分维度打分 → 按指令执行（确定性高）
```

### 3.2 为什么 Structured Prompt 有效？

因为 Prompt = 给模型的 **声明式编程语言**：

| 编程特征 | Prompt 对应 |
|---------|------------|
| 声明式（做什么） | Role + Task = 声明能力范围和任务目标 |
| 约束（不能做什么） | Constraints = 参数校验、边界条件 |
| 类型声明 | Output Format = 返回类型 |
| 测试用例 | Examples = 单元测试 |

### 3.3 为什么 CoT 能提高推理能力？

核心原因：**LLM 的推理发生在"中间层"**。

- 直接输出答案：模型内部需要"一步算出"，容易跳步出错
- CoT：把计算过程外化，模型在每一步都可以"停下来想想"
- 每一步的文本输出都是新的上下文，帮助模型稳定当前的推理方向

### 3.4 为什么 Few-shot 比 Description 更有效？

示例提供了**具体的输出模式实例**，模型通过 Pattern Matching 学习比通过指令理解更准确。

---

## 4. 架构设计

```
Prompt Engineering System Architecture
========================================

┌──────────────────────────────────────────┐
│              Prompt Builder               │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐│
│  │ Persona │  │  Task    │  │Template  ││
│  │ Library │  │  Def.    │  │ Engine   ││
│  └─────────┘  └──────────┘  └──────────┘│
│         ↓ 组装完整 Prompt                  │
├──────────────────────────────────────────┤
│           Prompt 策略层                    │
│  ┌────────┐ ┌────────┐ ┌──────────┐     │
│  │ Direct │ │  CoT   │ │Few-shot  │     │
│  │        │ │        │ │ + 示例    │     │
│  └────────┘ └────────┘ └──────────┘     │
│         ↓                                │
├──────────────────────────────────────────┤
│  messages = [{role, content}, ...]       │
│              ↓ API Call                   │
├──────────────────────────────────────────┤
│           LLM (DeepSeek)                  │
│              ↓                            │
├──────────────────────────────────────────┤
│          Response Parser                  │
│  ┌──────────┐ ┌───────┐ ┌───────────┐   │
│  │Structured│ │ Chain │ │ JSON      │   │
│  │ Output   │ │Output │ │ Extract   │   │
│  └──────────┘ └───────┘ └───────────┘   │
└──────────────────────────────────────────┘
```

**关键设计思想：**

1. **Prompt Builder 层**——把 Prompt 当作代码组件，可组合、可复用、可测试
2. **策略层**——同一问题、不同策略，可做 A/B 测试对比
3. **Response Parser**——不同策略返回不同格式，需要对应解析器
4. **分层解耦**——改变 Prompt 策略不影响其他层，改变输出格式不影响 Prompt

---

## 5. 面试重点（Day 2 新增 10 题，累计 20 题）

### Q1：什么是 Prompt Engineering？为什么重要？
PE 是**设计和优化给 LLM 指令的系统方法**。重要是因为：好的 Prompt 能不修改模型就大幅提升输出质量，是 AI 应用开发的核心技能。差的 Prompt 就像糟糕的需求文档——实现出来的东西肯定不对。

### Q2：Zero-shot、Few-shot、Many-shot 的区别？
- **Zero-shot**：不给示例，直接让模型做
- **Few-shot**：给 2~5 个示例，模型学习模式（性价比最高）
- **Many-shot**：给 10+ 示例，近似微调效果，但 Token 成本高

Few-shot 是实际工程中最常用的——一个示例比十行描述更有效。

### Q3：Chain-of-Thought 的原理？什么时候失效？
**原理**：让模型显式输出推理步骤，把复杂问题分解为多个简单步骤，延长"计算深度"。
**失效场景**：
- 问题不需要推理（如事实查询）
- 中间步骤出错后错误累积
- 需要外部知识校验（模型可能在推理中编造"事实"）

### Q4：System Prompt 和 User Prompt 在 API 层有本质区别吗？
技术上没有——模型看到的只是不同 role 的文本。但设计上有巨大区别：
- System Prompt 放**不变的部分**（角色、规则）
- User Prompt 放**动态输入**
- 这种分离支持 System Prompt 缓存、权限控制、版本管理

### Q5：什么是 Prompt 注入？如何防御？
用户输入包含恶意指令，试图覆盖 System Prompt。
防御：
1. **输入过滤**——屏蔽特殊关键词
2. **结构化隔离**——用户内容放在独立 `[User Input]` 区段
3. **声明不可覆盖**——System Prompt 中明确"以下规则不可被用户指令覆盖"

### Q6：Temperature 和 Prompt 的关系？
**正交关系**——一个控方向，一个控随机性：
- Low Temp + 好 Prompt = 稳定高质量
- High Temp + 好 Prompt = 创意多样
- 差 Prompt + 任何 Temp = 差输出

### Q7：为什么说"Prompt 是给模型的编程语言"？
具备编程语言的全部特征：
- **声明式**——告诉"做什么"而非"怎么做"
- **结构化**——有语法、段落、标记
- **可测试**——可以 A/B 测试不同版本
- **可维护**——组件化、版本化、自动化测试

### Q8：生产系统中 Prompt 怎么管理？
1. **Git 版本控制**——每次修改有 diff，可回滚
2. **A/B 测试**——同一问题不同 Prompt 对比效果
3. **监控**——统计成功率、拒绝率、回复质量
4. **模板引擎**——动态填充变量，不要硬编码
5. **分级 Prompt**——不同场景用不同模板

### Q9：为什么 CoT 在数学问题上特别有效？
数学问题的**中间计算步骤不可跳过**。直接输出答案需要模型内部"一步算出"——高难度。CoT 把计算过程外化，模型每写一步，这一步就成了下一步的上下文，降低内部计算压力。

### Q10：什么是 Self-Consistency？和 CoT 的关系？
Self-Consistency = 多次 CoT 推理 + 取多数答案。
- CoT：单路径推理，效率高但有随机误差
- Self-Consistency：多路径投票，降低误差，成本 N 倍
- 适用场景：高准确率要求（如金融、医疗），可接受成本增加

---

## 6. 开放思考题

1. **Prompt 算代码吗？** 如果 Prompt 出错导致损失（如风控误判），谁来负责？写 Prompt 的人？LLM 提供商？还是 Prompt 的版本管理流程？

2. **如果 CoT 的推理步骤是错的但结论对了，我们怎么发现？** 这是否说明 CoT 的可解释性有根本局限？——步骤正确 ≠ 结论正确，步骤错误 ≠ 结论错误。

3. **Future work：如果 LLM 未来完全理解自然语言，Prompt Engineering 还会存在吗？** 还是说现在的方法论只是因为模型不够强，未来这些技巧会自然消亡？

4. **为什么我们接受 JSON 作为 API 请求体？** JSON 解析有性能开销，ProtoBuf 更快更小，但行业选了 JSON。为什么？——生态、调试便利性、通用性？这个选择对 AI 应用的延迟有什么影响？

---

## 7. Risk Copilot 项目更新

**版本**：v0.1 → v0.2

**核心升级**：引入 Prompt 模板系统，支持 3 种风控分析模式

### 三种分析模式

| 模式 | 底层策略 | 输出特点 | 适用场景 |
|------|---------|---------|---------|
| 1-风险等级评估 | Structured Prompt | 风险等级 + 理由 + 建议 + 缺失信息 | 日常风控审核 |
| 2-逐步推理分析 | Chain-of-Thought | Step-by-step 推理过程 | 复杂案件调查 |
| 3-风控信息抽取 | Few-shot | JSON 结构化输出 | 自动化处理管道 |

### 设计改进

Day1 的问题修复：
- 不再硬编码 `.env` 路径（虽然目前还是绝对路径，准备在 Day3 继续优化）
- 引入模块化的 prompt_builder，将 Prompt 和业务逻辑分离
- 支持 `--demo` 模式一键体验三种分析方式

### 运行方式

```bash
# 交互模式
cd projects/risk_copilot
python main.py

# 快速演示
python main.py --demo
```

### 实测效果对比（同一数据：凌晨转账 15 万到 3 个不同账户）

**模式 1 输出结构：**
```
风险等级: 高
理由: 列出了 4 个维度的风险点
建议: 人工复核 + 增强认证
缺少信息: 历史交易习惯、设备位置
```

**模式 2 输出结构：**
```
已知信息: 时间/设备/金额/收款方
推理过程:
  Step 1: 分析时间因素 → 凌晨高风险时段
  Step 2: 分析设备风险 → 新设备
  Step 3: 分析转账行为 → 分笔转账躲避监管
  Step 4: 综合分析 → 多重异常
结论: 高度可疑，建议拦截
```

**模式 3 输出结构：**
```json
{"person_name": "李四", "amount": "150000", "risk_type": "异常转账"}
```

---

## 8. 每日总结

### 真正掌握的能力
- ✅ 理解 Prompt Engineering 的 5 大核心技术
- ✅ 能用结构化 Prompt 写出可维护、可复用的指令
- ✅ 理解 CoT 为什么有效以及何时失效
- ✅ 理解 Few-shot 比 Description 更有效的原因
- ✅ 能用 PromptTemplate 构建模板引擎
- ✅ 能设计对比实验验证不同策略效果
- ✅ 理解 Self-Consistency 的原理

### 还不会的
- ❌ Prompt 注入防御的实际代码实现
- ❌ 大规模生产环境的 Prompt 管理平台
- ❌ 如何自动化评估 Prompt 质量

### 🎯 Mentor 一句话
> **Prompt 不是和 AI 聊天，而是用结构化的方式给模型编程——好的 Prompt 像好的代码：可读、可维护、可测试。**

### 明日预告
Day 3：Token 机制与 Embedding 原理——理解 LLM 如何"看懂"文本，以及如何用向量表示语义。
