# Day 4：Vector Store 与 Context Grounding

**日期**：2026-07-21

---

## 1. 昨日 Review

Day3 完成了 Token / Embedding 的底层雏形：

- `tokenize()`：把文本拆成 token
- `hashing_embedding()`：把 token 序列映射成固定长度向量
- `cosine_similarity()`：计算两个文本向量的相似度
- `search_similar_cases()`：做 Top-K 相似案例检索
- `Risk Copilot v0.3`：新增相似案例检索模式

### Code Review

Day3 的代码能跑，但有三个关键问题：

- `hashing_embedding()` 只是教学版，不具备真实语义泛化能力。
- 相似案例只是被打印出来，没有变成 LLM 分析时的证据。
- 检索逻辑还不是一个稳定模块，缺少 document、metadata、context format 这些向量库基本抽象。

Day4 的任务就是补上这些工程抽象。

---

## 2. 今日目标

今天完成一个本地 `LocalVectorStore`，并把检索到的风险知识转换成 LLM 可使用的 grounded context。

---

## 3. Knowledge

### 3.1 Vector Store 是什么？

Vector Store 是向量数据库的最小抽象。它不是只存向量，而是存：

- `document id`
- `text`
- `embedding vector`
- `metadata`

真实系统中，一条知识通常长这样：

```text
doc_id: K001
text: 凌晨新设备大额转账属于高风险信号
vector: [0.12, -0.03, ...]
metadata: {"risk_type": "account_takeover", "severity": "high"}
```

### 3.2 为什么需要 metadata？

只靠向量相似度不够。

例如风控场景中，你可能只想查：

- 高风险案例
- 账户接管案例
- 最近 90 天案例
- 某个国家或渠道的案例

这时需要：

```text
vector similarity + metadata filter
```

也就是先限定业务范围，再做语义相似度排序。

### 3.3 什么是 Context Grounding？

Context Grounding 是把检索到的证据放进 Prompt，让 LLM 基于证据回答。

Day3：

```text
用户输入 → embedding → 找相似案例 → 打印结果
```

Day4：

```text
用户输入 → embedding → 找相似知识 → 格式化为 context → LLM 基于 context 分析
```

区别在于：Day4 开始让检索结果参与最终答案生成。

### 3.4 为什么不能直接把所有知识都塞给 LLM？

因为：

- 成本高：每次都传所有文档，token 成本爆炸
- 干扰多：无关上下文会降低回答质量
- 上下文有限：模型窗口不是无限的
- 难维护：知识库变化后 prompt 无法稳定控制

所以行业选择：

```text
先检索少量相关知识，再交给 LLM
```

这就是 RAG 的基本思想。

### 3.5 Grounded Answer 的核心约束

一个好的 grounded prompt 必须要求模型：

- 引用检索到的 `doc_id`
- 不编造 context 没有的事实
- context 不相关时要说明
- 区分“用户输入事实”和“历史知识依据”

这比简单说“请参考以下资料回答”更工程化。

---

## 4. Coding

### Step 1：定义 Document

文件：`demos/day04/local_vector_store.py`

```python
@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

这是向量库的最小数据单元。

### Step 2：实现 LocalVectorStore

```python
class LocalVectorStore:
    def add_documents(self, documents: list[Document]) -> None:
        ...

    def search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        ...
```

今天的版本是内存暴力检索，但接口形态和真实向量数据库类似。

### Step 3：格式化 Context

```python
def format_context(results: list[SearchResult]) -> str:
    ...
```

输出形态：

```text
[Context 1]
doc_id: K001
title: 凌晨新设备大额转账
score: 0.321
metadata: {...}
content: ...
```

### Step 4：构造 Grounded Prompt

```python
def build_grounded_prompt(query: str, results: list[SearchResult]) -> str:
    ...
```

关键约束：

- 必须引用至少一个 `doc_id`
- 如果 context 不相关，要明确说明
- 不要编造 context 中没有的事实

### Step 5：运行

```bash
cd D:\AI-Learning\AI-Engineer-Roadmap
python -X utf8 demos\day04\local_vector_store.py
```

Risk Copilot：

```bash
cd D:\AI-Learning\AI-Engineer-Roadmap\projects\risk_copilot
python -X utf8 main.py
```

选择模式 `5`：基于检索上下文分析。

---

## 5. Source Code Analysis

### 5.1 为什么向量库返回的不只是 text？

因为生产系统需要可追溯。

LLM 说“这是高风险”不够，必须知道依据来自哪里：

```text
doc_id: K001
score: 0.321
metadata: {"risk_type": "account_takeover"}
```

否则后面无法做审计、回放、评估和问题定位。

### 5.2 为什么要把 Context 格式化得很明确？

LLM 不会天然知道哪部分是证据、哪部分是用户输入。格式越明确，模型越不容易混淆：

```text
## Retrieved Context
...

## User Input
...

## Constraints
...
```

这是 prompt engineering 和 retrieval engineering 的交汇点。

### 5.3 为什么还没有接真实向量数据库？

因为现在的目标是理解核心抽象：

```text
Document → Embedding → Vector Store → Search Result → Context → Prompt
```

等这个链路清楚后，再换成 FAISS、Chroma、Milvus、pgvector，只是替换存储和检索实现。

---

## 6. Architecture

```text
Risk Copilot v0.4
=================

User Risk Text
    ↓
LocalVectorStore.search()
    ↓
Top-K Risk Documents
    ↓
format_context()
    ↓
build_grounded_prompt()
    ↓
LLM
    ↓
Grounded Risk Analysis
```

今天新增的是：

```text
Top-K Results → Context Formatting → Grounded Prompt
```

这一步是从“检索 demo”走向“AI 应用”的关键。

---

## 7. Interview

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

### 追加追问

**Q11：Grounded Prompt 的 Constraints 写得很具体，LLM 一定会遵守吗？**
不一定会。Constraint 是弱约束（文本描述），不是强约束（编译检查）。生产系统需要额外做输出校验——检查 LLM 是否真的引用了 doc_id，没有引用则重试或降级到规则引擎。

**Q12：风控场景用 Cosine Similarity 还是 Dot Product？**
如果 embedding 做了 L2 Normalization（你的 hashing_embedding 做了），cosine = dot product。如果没做，必须用 cosine。生产系统的 embedding 通常已归一化，两者等价。

**Q13：检索结果语义相似但不相关怎么办？**
这是 RAG 经典问题。解决方案：① Reranker 重排序 ② Metadata Filter 先过滤 ③ Prompt 要求模型判断相关性（你的 Constraint 已经做到了——实际测试中 LLM 正确排除了 K002/K003）

**Q14：5 条知识库扩展到 1000 条会怎样？**
暴力检索 O(N) 复杂度。1000 条没问题，1 万条开始卡。生产用 ANN 索引（HNSW O(log N)）。

**Q15：Embedding 存储用什么数据结构？**
内存中可用 HashMap，生产用 HNSW graph 或 IVF 倒排索引。FAISS、Chroma 等库帮你管理。

---

## 8. Thinking

1. 如果检索到的 context 和用户输入相似但结论相反，LLM 应该如何处理？
2. 风控系统中，context 引用是解释能力，还是合规必需品？
3. 如果 Top1 错了，但 Top2 对了，Prompt 应该如何设计才能降低误导？
4. 什么时候应该相信规则引擎，什么时候应该相信 RAG + LLM？
5. 你的 hashing_embedding 中相同 token 永远映射到相同 bucket——真实 embedding model 有没有这个性质？为什么？
6. 模式 5 的 grounded answer 正确排除了 K002/K003——这是好事。但如果 context 和用户输入直接冲突（用户说"没转账"，context 说"已转出"），模型该信谁？
3. 如果 Top1 错了，但 Top2 对了，Prompt 应该如何设计才能降低误导？
4. 什么时候应该相信规则引擎，什么时候应该相信 RAG + LLM？

---

## 9. 今日总结

### 今天真正掌握的能力

- 理解 Vector Store 的核心抽象
- 能实现本地内存向量库
- 能用 metadata filter 缩小检索范围
- 能把检索结果格式化为 LLM context
- 能设计 grounded prompt 约束模型基于证据回答
- 理解 RAG 的工程链路雏形

### 还不会的

- 还没有接真实 embedding API
- 还没有接 FAISS、Chroma、Milvus 或 pgvector
- 还没有做 chunking 策略
- 还没有做 RAG 评估集

---

## 10. Risk Copilot 项目更新

**版本**：v0.3 → v0.4

新增：

- `demos/day04/local_vector_store.py`
- `Risk Copilot` 新增模式 5：基于检索上下文分析

今天的一句话：

> 检索不是终点，把证据组织成 LLM 能遵守的上下文，才是 RAG 工程的开始。
