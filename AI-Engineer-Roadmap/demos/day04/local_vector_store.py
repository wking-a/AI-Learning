"""
Day 4: Local Vector Store and Context Grounding
===============================================
功能：
  1. 用本地向量库管理风控知识片段
  2. 支持 Top-K 相似检索和 metadata filter
  3. 把检索结果格式化成可放入 LLM Prompt 的上下文

说明：
  Day3 只做了相似案例检索。
  Day4 开始把检索结果变成“可被 LLM 使用的证据上下文”。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float


def tokenize(text: str) -> list[str]:
    """Teaching tokenizer: Chinese chars, English words, numbers, and symbols."""
    return TOKEN_PATTERN.findall(text.lower())


def hashing_embedding(text: str, dim: int = 128) -> list[float]:
    """A deterministic local embedding for learning the vector-store pipeline."""
    vector = [0.0] * dim

    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LocalVectorStore:
    """A tiny in-memory vector store that mirrors real vector DB concepts."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._documents: list[Document] = []
        self._vectors: dict[str, list[float]] = {}

    def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self._documents.append(document)
            self._vectors[document.doc_id] = hashing_embedding(document.text, dim=self.dim)

    def search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_vector = hashing_embedding(query, dim=self.dim)
        results = []

        for document in self._documents:
            if metadata_filter and not self._matches_filter(document, metadata_filter):
                continue

            score = cosine_similarity(query_vector, self._vectors[document.doc_id])
            results.append(SearchResult(document=document, score=score))

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _matches_filter(document: Document, metadata_filter: dict[str, Any]) -> bool:
        return all(document.metadata.get(key) == value for key, value in metadata_filter.items())


RISK_KNOWLEDGE_BASE = [
    Document(
        doc_id="K001",
        title="凌晨新设备大额转账",
        text="凌晨时段使用新设备登录，并向新添加收款人发起大额转账，通常属于账户接管或盗刷高风险信号。",
        metadata={"risk_type": "account_takeover", "severity": "high"},
    ),
    Document(
        doc_id="K002",
        title="钓鱼短信验证码泄露",
        text="短信声称银行卡异常，诱导用户点击链接并填写身份证、银行卡号或验证码，常见于钓鱼攻击。",
        metadata={"risk_type": "phishing", "severity": "high"},
    ),
    Document(
        doc_id="K003",
        title="安全账户诈骗",
        text="冒充客服、公安或平台人员，要求用户把资金转入所谓安全账户，是典型电信诈骗手法。",
        metadata={"risk_type": "scam", "severity": "high"},
    ),
    Document(
        doc_id="K004",
        title="异地登录后解绑",
        text="账户在异地 IP 登录后立刻修改密码、解绑银行卡或关闭通知，可能说明账户控制权发生异常变化。",
        metadata={"risk_type": "account_takeover", "severity": "medium"},
    ),
    Document(
        doc_id="K005",
        title="常用设备小额消费",
        text="用户在常用设备、常用地点、常规时段发生小额消费，通常属于低风险正常行为。",
        metadata={"risk_type": "normal_payment", "severity": "low"},
    ),
]


def build_store() -> LocalVectorStore:
    store = LocalVectorStore()
    store.add_documents(RISK_KNOWLEDGE_BASE)
    return store


def format_context(results: list[SearchResult]) -> str:
    """Format retrieved documents as grounded context for an LLM prompt."""
    context_blocks = []

    for index, result in enumerate(results, start=1):
        document = result.document
        context_blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"doc_id: {document.doc_id}",
                    f"title: {document.title}",
                    f"score: {result.score:.3f}",
                    f"metadata: {document.metadata}",
                    f"content: {document.text}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


def build_grounded_prompt(query: str, results: list[SearchResult]) -> str:
    context = format_context(results)
    return f"""## Role
你是金融风控分析师。

## Task
基于检索到的历史风险知识，分析用户输入的风险等级。

## Retrieved Context
{context}

## User Input
{query}

## Constraints
- 必须引用至少一个 context 的 doc_id
- 如果 context 与用户输入不相关，要明确说明
- 不要编造 context 中没有的事实

## Output Format
风险等级: [低/中/高/极高]
引用依据: [doc_id + 原因]
分析: [结合用户输入和上下文]
建议: [处置建议]
"""


def demo() -> None:
    query = "用户凌晨换新手机登录，准备向刚添加的陌生账户转账6万元。"
    store = build_store()
    results = store.search(query, top_k=3)

    print("=" * 70)
    print("Day 4: Local Vector Store Search")
    print("=" * 70)
    print(f"Query: {query}")

    for rank, result in enumerate(results, start=1):
        document = result.document
        print(f"\nTop {rank} | score={result.score:.3f} | {document.doc_id}")
        print(f"title: {document.title}")
        print(f"metadata: {document.metadata}")
        print(document.text)

    print("\n" + "=" * 70)
    print("Day 4: Grounded Prompt")
    print("=" * 70)
    print(build_grounded_prompt(query, results))


if __name__ == "__main__":
    demo()
