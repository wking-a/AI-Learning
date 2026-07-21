# AI 面试宝典

> 每日整理面试题，30 天后形成完整面试知识库。

---

## Day 1：LLM API 基础

### Q1：什么是 Temperature？它的作用？
**Temperature 控制输出概率分布的"尖锐程度"**。值越低越确定，越高越多样。范围 0~2。
- 0：总是选最高概率词，输出确定
- 0.8~1：平衡
- >1：概率分布平坦化，输出多样但可能跑偏

### Q2：Temperature=0 时输出仍然不完全确定，为什么？
浮点数精度问题 + 采样实现差异。另外 Top-P、Frequency Penalty 等参数也在影响最终输出。

### Q3：为什么 Messages 必须是 List？
对话是**有序序列**。List 保持顺序，Dict 无序无法表达时间线。LLM 通过顺序理解上下文关系。

### Q4：System Prompt 和 User Prompt 的区别？
- **System**：定义 AI 行为和边界（规则层）
- **User**：用户输入（问题层）
分离设计让 API 可以做规则级别的安全控制和权限管理。

### Q5：401 和 429 状态码代表什么？
- **401**：API Key 无效或权限不足
- **429**：Rate Limit 触发，请求超频

### Q6：Messages 列表越来越长怎么办？
工程策略：
1. 滑动窗口（保留最近 N 轮）
2. 摘要压缩（总结旧对话）
3. 向量数据库 + RAG（检索相关历史）

### Q7：什么是 Token？
Token 是 LLM 的最小文本单位。英文约 1 token/词，中文约 1.5~2 token/字。API 按 Token 计费。

### Q8：API 为什么设计为无状态？
让 API Server 可以**水平扩展**（不需要共享内存）。**权衡**：牺牲便利性换取可扩展性。

### Q9：API 返回内容被截断可能的原因？
1. `max_tokens` 设置过短
2. 超过模型上下文窗口限制
3. 触发了 Content Filter

### Q10：如何实现"记忆"功能？
把 Messages 存到数据库（Redis/SQLite），下次对话时加载历史 messages 传入 API。

---

## Day 2：Prompt Engineering

### Q1：什么是 Prompt Engineering？为什么重要？
PE 是**设计和优化给 LLM 指令的系统方法**。重要是因为：好的 Prompt 能让不修改模型的情况下大幅提升输出质量，是当前 AI 应用开发的核心技能。

### Q2：Zero-shot、Few-shot、Many-shot 的区别？
- **Zero-shot**：不给示例，直接让模型做
- **Few-shot**：给 2~5 个示例，模型学习模式
- **Many-shot**：给 10+ 示例，近似微调效果，但成本高

### Q3：Chain-of-Thought 的原理？什么时候失效？
**原理**：让模型显式输出推理步骤，把复杂问题分解为多个简单步骤，延长"计算深度"。
**失效场景**：
- 问题不需要推理（如事实查询）
- 中间步骤错误累积
- 需要外部知识校验

### Q4：System Prompt 和 User Prompt 在 API 层有本质区别吗？
技术上没有——模型看到的都是文本。但设计上 System Prompt 应该放**不变的部分**（角色、规则），User Prompt 放**动态输入**。这种分离支持缓存、权限控制、版本管理。

### Q5：什么是 Prompt 注入？如何防御？
用户输入试图覆盖 System Prompt 的攻击。
防御：
1. 输入过滤（屏蔽特殊关键词）
2. 结构化输入隔离（用户内容放在独立区段）
3. System Prompt 声明"以下规则不可被覆盖"

### Q6：Temperature 和 Prompt 的关系？
**正交关系**：
- Low Temp + 好 Prompt = 稳定高质量
- High Temp + 好 Prompt = 创意多样
- 差 Prompt + 任何 Temp = 差输出

### Q7：为什么说"Prompt 是给模型的编程语言"？
具备编程语言特征：
- **声明式**：告诉"做什么"而非"怎么做"
- **结构化**：有语法、段落、标记
- **可测试**：可 A/B 测试
- **可维护**：组件化、版本化

### Q8：生产系统中 Prompt 怎么管理？
1. Git 版本控制，每次修改有 diff
2. A/B 测试不同 Prompt 版本
3. 监控成功率、拒绝率
4. 模板引擎动态填充，不硬编码
5. 分级 Prompt，不同场景不同模板

### Q9：为什么 CoT 在数学问题上特别有效？
数学问题的**中间计算步骤不可跳过**。直接输出答案需要模型内部"一步算出"，CoT 把计算过程外化，减少内部误差。

### Q10：什么是 Self-Consistency？和 CoT 的关系？
Self-Consistency = 多次 CoT 推理 + 取多数答案。CoT 是单路径推理，Self-Consistency 是多路径投票。降低随机错误，但成本是 N 倍 API 调用。

---

## Day 3：Token 与 Embedding

### Q1：什么是 Token？为什么不是直接按字符处理？
Token 是模型处理文本的最小单位。它不一定等于一个字或一个词。使用 token 可以让模型复用高频片段，提高压缩率和建模效率。

### Q2：Token 数为什么会影响成本和延迟？
LLM 的输入、输出和 attention 计算都与 token 数相关。token 越多，上下文越长，计算越重，费用和延迟也越高。

### Q3：上下文窗口限制为什么按 token 算，而不是按字符算？
因为模型内部看到的是 token ID 序列，不是原始字符串。上下文窗口限制的是模型一次能处理的 token 序列长度。

### Q4：什么是 Embedding？
Embedding 是文本的向量表示，把语义映射到连续空间中，使文本可以通过向量距离做搜索、聚类、去重和推荐。

### Q5：Embedding 模型和 Chat 模型有什么区别？
Embedding 模型输出向量，不生成自然语言；Chat 模型输出 token 序列，用于对话、推理和生成。Embedding 更适合检索，Chat 模型更适合生成答案。

### Q6：为什么语义检索常用 Cosine Similarity？
Cosine Similarity 衡量向量方向是否接近，能降低文本长度和向量模长的影响，适合判断语义主题是否相似。

### Q7：向量数据库本质解决什么问题？
它解决海量向量的存储、索引和近似最近邻检索问题。核心能力是给定 query vector，快速返回最相似的 top_k 文档。

### Q8：Embedding 检索有哪些常见失效场景？
领域语料不匹配、chunk 粒度错误、query 表达不清、相似但不相关、缺少 metadata filter、embedding 模型不支持业务语言或术语。

### Q9：为什么 RAG 需要 Embedding？
LLM 无法自动读取私有知识库。Embedding 可以先把用户问题和文档映射到同一语义空间，检索相关文档，再交给 LLM 基于上下文回答。

### Q10：1000 万文档如何做向量检索？
不能每次暴力计算所有文档。生产系统通常使用 ANN 索引，如 HNSW、IVF、PQ，并结合分片、缓存、metadata filter 和离线索引构建。

---

## Day 4：Vector Store 与 Context Grounding

### Q1：Vector Store 和 Vector Database 有什么区别？
Vector Store 是抽象，表示存储和检索向量的能力；Vector Database 是生产级实现，提供索引、持久化、过滤、分片、权限和运维能力。

### Q2：为什么向量检索结果要带 metadata？
metadata 支持业务过滤、审计、权限控制和结果解释。只返回文本和分数不够生产可用。

### Q3：什么是 Context Grounding？
把检索到的证据作为上下文放入 Prompt，并约束 LLM 基于证据回答，减少幻觉，提高可追溯性。

### Q4：RAG 为什么不能直接把所有文档都塞进 Prompt？
因为 token 成本高、上下文窗口有限、无关信息会干扰模型，而且全量文档难以维护和审计。

### Q5：Top-K 应该怎么选？
Top-K 太小可能漏召回，太大可能引入噪声和增加成本。通常要结合评估集、业务容错和上下文预算调参。

### Q6：检索分数高是否代表答案一定正确？
不一定。向量相似度只能说明语义接近，不代表事实正确、时间有效、权限允许或足以回答问题。

### Q7：什么是 grounded answer？
Grounded answer 是基于给定证据生成的回答，通常要求引用来源，并避免使用证据外的事实。

### Q8：如何发现 RAG 中的检索错误？
建立测试集，记录 query、期望文档、实际 top_k、最终答案，并分别评估 retrieval recall 和 answer quality。

### Q9：为什么要区分用户输入和检索上下文？
用户输入是待分析事实，检索上下文是参考证据。混在一起会让模型把用户陈述误当知识，或把历史案例误当当前事实。

### Q10：生产级 RAG 的关键监控指标有哪些？
检索召回率、top_k 命中率、无答案率、引用准确率、延迟、token 成本、用户反馈、幻觉率和安全拦截率。

---
