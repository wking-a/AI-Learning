# Day 3：Token 机制与 Embedding 原理

**日期**：2026-07-10

---

## 1. 昨日 Review

Day2 已经完成 Prompt Engineering 的工程化雏形：

- `PromptTemplate`：把 Prompt 拆成 Role / Task / Constraints / Output Format / Examples
- `experiment.py`：对比 Naive、Structured Prompt、CoT、Few-shot、多维评估
- `Risk Copilot v0.2`：支持三种风控分析模式

### Code Review

当前代码能跑，但有几个工程问题：

- `experiment.py` 和 `risk_copilot/main.py` 仍然硬编码 `.env` 与 `day02` 路径，后续应抽成统一配置。
- `PromptTemplate.build()` 使用 `str.format()`，如果用户输入里包含 `{}`，可能触发格式化异常。
- Day2 的 Few-shot 输出只是“要求 JSON”，还没有真正做 JSON parse 和 schema 校验。
- `Risk Copilot` 目前是 CLI demo，状态、配置、模型调用、业务逻辑还混在一个文件里。

这些问题不是 Day3 全部修完，因为今天重点是 Token / Embedding。我们先把“相似案例检索”这个能力接起来。

---

## 2. 今日目标

今天完成一个可运行的 Token + Embedding Demo，并把 Risk Copilot 升级为支持“相似历史案例检索”的 v0.3。

---

## 3. Knowledge

### 3.1 什么是 Token？

Token 是模型处理文本的最小单位。它不一定等于一个字，也不一定等于一个词。

例如：

```text
用户半夜转账50000元
```

可能会被拆成：

```text
["用", "户", "半", "夜", "转", "账", "50000", "元"]
```

真实 tokenizer 会更复杂，可能把高频词合并成一个 token，也可能把罕见词拆得更细。

### 3.2 为什么 LLM 不直接处理字符串？

模型不能直接理解 Python 字符串。神经网络处理的是数字张量，所以文本进入模型前必须经过：

```text
Text
  ↓ tokenizer
Token IDs
  ↓ embedding table
Vectors
  ↓ transformer
Next token probability
```

Token 是文本和神经网络之间的接口。

### 3.3 Token 为什么影响成本和上下文？

API 计费通常按 token 算：

- 输入 token：system prompt、history、user prompt、RAG context
- 输出 token：模型生成的回复

上下文窗口也是 token 限制，不是字符限制。

这意味着：

- Prompt 写得越长，成本越高。
- 历史对话越长，延迟越高。
- RAG 检索塞太多文档，会挤占用户问题和模型输出空间。

### 3.4 什么是 Embedding？

Embedding 是把文本映射成固定长度向量。

```text
"凌晨新设备大额转账"
  ↓ embedding model
[0.12, -0.03, 0.77, ..., 0.21]
```

Embedding 的核心价值不是“压缩文本”，而是把语义变成可以计算的几何空间。

语义相近的文本，向量距离更近：

```text
"凌晨新设备转账"        ≈ "半夜换手机转账"
"钓鱼短信验证码"        ≈ "异常短信要求填写验证码"
"白天咖啡消费28元"      ≠ "异地登录后解绑银行卡"
```

### 3.5 为什么用 Cosine Similarity？

Cosine Similarity 衡量两个向量方向是否接近。

```text
similarity = dot(a, b) / (|a| * |b|)
```

在语义检索中，我们通常更关心“方向”而不是“长度”。长度可能受文本长短影响，方向更接近语义主题。

### 3.6 Embedding 和 LLM 的关系

Embedding 不是 ChatGPT，也不是一个会回答问题的模型。它只负责：

- 把文本转向量
- 支持相似度计算
- 支持检索、聚类、去重、推荐

LLM 负责生成答案；Embedding 负责找到相关信息。

这就是 RAG 的基础：

```text
User Query
  ↓ embedding
Vector Search
  ↓ retrieve documents
LLM
  ↓ answer grounded by documents
```

---

## 4. Coding

### Step 1：Tokenize

文件：`demos/day03/token_embedding_demo.py`

实现一个最小 tokenizer：

```python
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
```

它把中文字符、英文数字词、符号拆开，方便观察 token 数量。

### Step 2：Embedding

为了不依赖外部 API，今天先写一个 deterministic hashing embedding：

```python
def hashing_embedding(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    ...
```

真实 embedding model 是训练出来的；这里的 hashing embedding 只是让检索流程本地可运行。

### Step 3：Similarity Search

```python
def search_similar_cases(query: str, cases: list[Case], top_k: int = 3):
    query_vector = hashing_embedding(query)
    scored_cases = []

    for case in cases:
        case_vector = hashing_embedding(case.text)
        score = cosine_similarity(query_vector, case_vector)
        scored_cases.append((case, score))

    return sorted(scored_cases, key=lambda item: item[1], reverse=True)[:top_k]
```

这就是向量数据库最核心的抽象：

```text
query → vector → compare with document vectors → top_k
```

### Step 4：运行

```bash
cd D:\AI-Learning\AI-Engineer-Roadmap
python demos\day03\token_embedding_demo.py
```

Risk Copilot：

```bash
cd D:\AI-Learning\AI-Engineer-Roadmap\projects\risk_copilot
python main.py --demo
```

---

## 5. Source Code Analysis

### 5.1 为什么 embedding 一定是固定长度？

因为后续要做矩阵计算和向量索引。不同长度的文本如果输出不同长度向量，就无法统一存储、批量计算、建索引。

### 5.2 向量数据库本质做了什么？

向量数据库不是神秘组件，它主要做三件事：

- 存储：保存 `id -> vector + metadata`
- 检索：给定 query vector，找最近的 top_k
- 索引：用 HNSW、IVF、PQ 等算法加速近邻搜索

今天我们手写的是最小暴力检索版：每次 query 都和所有案例算相似度。

### 5.3 为什么真实系统不用暴力检索？

如果只有 5 个案例，暴力检索没问题。

如果有 1000 万文档：

```text
每次 query 都算 1000 万次 cosine similarity
```

延迟和成本都不可接受。所以生产系统会用 ANN（Approximate Nearest Neighbor）索引，用少量精度损失换速度。

---

## 6. Architecture

```text
Risk Copilot v0.3
=================

User Risk Text
    ↓
Tokenizer
    ↓
Embedding Encoder
    ↓
Vector Similarity Search
    ↓
Top-K Similar Cases
    ↓
Risk Analyst / LLM Prompt
    ↓
Risk Decision
```

今天新增的是中间这层：

```text
Embedding Encoder → Vector Similarity Search → Top-K Similar Cases
```

它是 Day8 RAG 的前置基础。

---

## 7. Interview

### Q1：什么是 Token？为什么不是直接按字符处理？

Token 是模型处理文本的最小单位。不是直接按字符，是因为 BPE / SentencePiece 等 tokenizer 会把高频片段合并，提高压缩率和建模效率。

### Q2：Token 数为什么会影响成本？

LLM API 的计算量和输入输出 token 数相关。更多 token 意味着更长上下文、更大 attention 计算、更高延迟和费用。

### Q3：中文和英文 token 计数有什么差异？

英文常见词可能是一个 token，中文通常更接近按字或短词切分。具体取决于 tokenizer 训练语料和算法。

### Q4：Embedding 是什么？

Embedding 是文本的向量表示，把语义映射到连续空间中，使相似度、检索、聚类等操作可以通过数学计算完成。

### Q5：Embedding 和 LLM 生成模型有什么区别？

Embedding model 输出向量，不生成自然语言；LLM 生成 token 序列。Embedding 常用于检索，LLM 用于理解和生成。

### Q6：为什么语义检索常用 cosine similarity？

因为 cosine 衡量向量方向相似性，能减少文本长度和向量模长的影响，更适合比较语义方向。

### Q7：向量数据库解决什么问题？

解决海量向量的存储、索引和近似最近邻检索问题。核心是快速找到与 query vector 最接近的 top_k 文档。

### Q8：Embedding 检索一定准确吗？

不一定。它可能受模型质量、语料领域、chunk 粒度、查询表达、相似度算法影响。生产中通常需要 reranker、过滤条件和评估集。

### Q9：为什么 RAG 需要 Embedding？

因为 LLM 本身不会自动访问你的私有知识库。Embedding 可以把 query 和文档放进同一语义空间，先检索相关文档，再交给 LLM 生成答案。

### Q10：1000 万文档如何做向量检索？

通常使用向量数据库或 ANN 索引，如 HNSW、IVF、PQ。还要做 metadata filter、分片、缓存、冷热数据分层、离线重建索引。

---

## 8. Thinking

1. 如果两个文本词面完全不同，但语义相同，hashing embedding 能处理好吗？真实 embedding 为什么可以？
2. 向量相似度高，是否一定代表答案相关？
3. 风控场景中，应该更相信“历史相似案例”还是“LLM 的推理解释”？
4. 如果检索结果错了，LLM 会不会基于错误上下文给出更自信的错误答案？

---

## 9. 今日总结

### 今天真正掌握的能力

- 理解 Token 是 LLM 的文本接口
- 理解 Token 会影响成本、延迟和上下文窗口
- 理解 Embedding 是语义向量，不是生成模型
- 能手写最小 embedding 检索流程
- 能用 cosine similarity 做 Top-K 相似案例检索
- 知道向量数据库和 RAG 的底层雏形

### 还不会的

- 还没有接真实 embedding API
- 还没有使用向量数据库
- 还没有做 chunking、metadata filter、reranker
- 还没有建立检索效果评估集

---

## 10. Risk Copilot 项目更新

**版本**：v0.2 → v0.3

新增：

- `demos/day03/token_embedding_demo.py`
- `Risk Copilot` 新增模式 4：相似案例检索

今天的一句话：

> Embedding 的价值不是让模型“回答问题”，而是让文本可以被搜索、比较和组织。
