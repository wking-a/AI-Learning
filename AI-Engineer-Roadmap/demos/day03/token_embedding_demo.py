"""
Day 3: Token and Embedding Demo
===============================
功能：
  1. 观察文本如何被拆成 token
  2. 用一个最小可运行的 hashing embedding 表示语义
  3. 基于 cosine similarity 做相似风控案例检索

说明：
  这个 Demo 不追求替代真实 embedding model。
  它的目标是让你看懂 Embedding + 相似度检索的工程骨架。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


@dataclass
class Case:
    case_id: str
    title: str
    text: str
    label: str


RISK_CASES = [
    Case(
        case_id="C001",
        title="凌晨新设备大额转账",
        text="用户在凌晨2点使用新手机登录，向新添加收款人转账50000元。",
        label="高风险",
    ),
    Case(
        case_id="C002",
        title="钓鱼短信诱导点击",
        text="客户收到短信称银行卡异常，要求点击链接填写身份证和验证码。",
        label="高风险",
    ),
    Case(
        case_id="C003",
        title="常用设备小额消费",
        text="用户使用常用手机在白天购买咖啡，支付金额28元。",
        label="低风险",
    ),
    Case(
        case_id="C004",
        title="客服诈骗安全账户",
        text="自称平台客服来电，要求用户把资金转入所谓安全账户。",
        label="高风险",
    ),
    Case(
        case_id="C005",
        title="异地登录后修改密码",
        text="账户在异地IP登录后立刻修改密码，并尝试解绑银行卡。",
        label="中高风险",
    ),
]


def tokenize(text: str) -> list[str]:
    """A tiny tokenizer for demonstration: Chinese chars, words, and symbols."""
    return TOKEN_PATTERN.findall(text.lower())


def token_stats(text: str) -> dict[str, int]:
    tokens = tokenize(text)
    return {
        "chars": len(text),
        "tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
    }


def hashing_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Convert tokens into a fixed-size vector.

    Real embedding models learn semantic vectors from data. This demo uses a
    deterministic hash so the retrieval pipeline can run locally without API keys.
    """
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


def search_similar_cases(query: str, cases: list[Case], top_k: int = 3) -> list[tuple[Case, float]]:
    query_vector = hashing_embedding(query)
    scored_cases = []

    for case in cases:
        case_vector = hashing_embedding(case.text)
        score = cosine_similarity(query_vector, case_vector)
        scored_cases.append((case, score))

    return sorted(scored_cases, key=lambda item: item[1], reverse=True)[:top_k]


def print_token_report(text: str) -> None:
    tokens = tokenize(text)
    stats = token_stats(text)

    print("=" * 70)
    print("[Token Report]")
    print("=" * 70)
    print(f"文本: {text}")
    print(f"字符数: {stats['chars']}")
    print(f"Token 数: {stats['tokens']}")
    print(f"去重 Token 数: {stats['unique_tokens']}")
    print(f"Tokens: {tokens}")


def print_embedding_report(text: str) -> None:
    vector = hashing_embedding(text, dim=16)
    preview = ", ".join(f"{value:.3f}" for value in vector)

    print("\n" + "=" * 70)
    print("[Embedding Report]")
    print("=" * 70)
    print("Embedding 是固定长度向量。这里展示 16 维 preview：")
    print(f"[{preview}]")


def print_search_report(query: str) -> None:
    print("\n" + "=" * 70)
    print("[Similar Case Search]")
    print("=" * 70)
    print(f"Query: {query}")

    for rank, (case, score) in enumerate(search_similar_cases(query, RISK_CASES), start=1):
        print(f"\nTop {rank} | score={score:.3f} | {case.label}")
        print(f"{case.case_id} - {case.title}")
        print(case.text)


def main() -> None:
    query = "用户半夜换了新手机登录，并准备给刚添加的陌生账户转账6万元。"

    print_token_report(query)
    print_embedding_report(query)
    print_search_report(query)


if __name__ == "__main__":
    main()
