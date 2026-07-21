"""
Risk Copilot - Day 4
====================
升级：引入本地向量库和检索上下文，支持 Grounded Risk Analysis

已有功能：
  1. 风险等级评估（Structured）
  2. 逐步推理分析（CoT）
  3. 信息抽取（Few-shot）
  4. 相似案例检索（Embedding + Cosine Similarity）
新增功能：
  5. 基于检索上下文的风控分析（Context Grounding）
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# 复用 Day2 的 Prompt Builder
sys.path.insert(0, r"d:\AI-Learning\AI-Engineer-Roadmap\demos\day02")
from prompt_builder import (
    risk_analysis_template,
    chain_of_thought_template,
    extractor_template,
)

# 复用 Day3 的最小 Embedding 检索 Demo
sys.path.insert(0, r"d:\AI-Learning\AI-Engineer-Roadmap\demos\day03")
from token_embedding_demo import RISK_CASES, search_similar_cases

# 复用 Day4 的本地向量库和上下文构造
sys.path.insert(0, r"d:\AI-Learning\AI-Engineer-Roadmap\demos\day04")
from local_vector_store import build_grounded_prompt, build_store, format_context

load_dotenv(r"d:\AI-Learning\AI-Engineer-Roadmap\demos\day01\.env")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


# ── 系统 Prompt ──────────────────────────────
SYSTEM_PROMPT = """你是一个严格按照用户指令执行任务的 AI 助手。不要偏离用户给出的分析框架。"""


# ── 分析模式 ─────────────────────────────────
MODES = {
    "1": {
        "name": "风险等级评估",
        "desc": "使用结构化 Prompt 进行多维度风险评估",
        "builder": risk_analysis_template,
    },
    "2": {
        "name": "逐步推理分析",
        "desc": "使用 Chain-of-Thought 逐步推理",
        "builder": chain_of_thought_template,
    },
    "3": {
        "name": "风控信息抽取",
        "desc": "从文本中抽取风控关键字段（Few-shot）",
        "builder": extractor_template,
    },
    "4": {
        "name": "相似案例检索",
        "desc": "用 Embedding + Cosine Similarity 查找历史相似案例",
        "builder": None,
    },
    "5": {
        "name": "基于检索上下文分析",
        "desc": "先检索风险知识，再让 LLM 基于证据分析",
        "builder": None,
    },
}


def call_llm(system: str, user: str, temp: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
    )
    return response.choices[0].message.content


def print_similar_cases(text: str) -> None:
    """Retrieve similar historical risk cases before using a real vector DB."""
    print("\n[相似案例]")
    for rank, (case, score) in enumerate(search_similar_cases(text, RISK_CASES), start=1):
        print(f"\nTop {rank} | score={score:.3f} | {case.label}")
        print(f"{case.case_id} - {case.title}")
        print(case.text)


def build_context_grounded_prompt(text: str) -> str:
    store = build_store()
    results = store.search(text, top_k=3)
    return build_grounded_prompt(text, results)


def print_retrieved_context(text: str) -> None:
    store = build_store()
    results = store.search(text, top_k=3)

    print("\n[检索上下文]")
    print(format_context(results))


def main():
    print("=" * 60)
    print("Risk Copilot v0.4 - Context Grounding")
    print(f"模型: {MODEL}")
    print("=" * 60)
    print("\n请选择分析模式：")
    for key, mode in MODES.items():
        print(f"  [{key}] {mode['name']} - {mode['desc']}")
    print("  [q] 退出")

    while True:
        choice = input("\n选择模式: ").strip()
        if choice.lower() == "q":
            break

        mode = MODES.get(choice)
        if not mode:
            print("无效选择，请重试")
            continue

        text = input("输入待分析文本: ").strip()
        if not text:
            print("文本不能为空")
            continue

        if choice == "4":
            print_similar_cases(text)
            continue

        if choice == "5":
            print_retrieved_context(text)
            grounded_prompt = build_context_grounded_prompt(text)
            print(f"\n[分析中...] 使用 {mode['name']}")
            try:
                reply = call_llm(SYSTEM_PROMPT, grounded_prompt)
                print(f"\n{reply}\n")
            except Exception as e:
                print(f"[ERROR] {e}")
            continue

        # 构建 Prompt
        template = mode["builder"]()
        user_prompt = template.build()
        user_prompt += f"\n\n## Data to Analyze\n{text}"

        print(f"\n[分析中...] 使用 {mode['name']}")
        try:
            reply = call_llm(SYSTEM_PROMPT, user_prompt)
            print(f"\n{reply}\n")
        except Exception as e:
            print(f"[ERROR] {e}")

    print("再见！")


# ── 快速演示 ─────────────────────────────────
def demo():
    """快速展示三种模式的效果对比"""
    test_data = "用户李四在凌晨1点使用新手机登录，尝试分3笔共转账15万元到3个不同银行账户。"

    print("=" * 60)
    print("Risk Copilot v0.4 - 快速演示")
    print("测试数据: " + test_data)
    print("=" * 60)

    for key, mode in MODES.items():
        print(f"\n{'─' * 60}")
        print(f"[{key}] {mode['name']}")
        print(f"{'─' * 60}")

        if key == "4":
            print_similar_cases(test_data)
            continue

        if key == "5":
            grounded_prompt = build_context_grounded_prompt(test_data)
            print("\n[Grounded Prompt]")
            print(grounded_prompt)
            print(f"\n[分析中...] 使用 {mode['name']}")
            try:
                reply = call_llm(SYSTEM_PROMPT, grounded_prompt)
                print(f"\n{reply}\n")
            except Exception as e:
                print(f"[ERROR] {e}")
            continue

        template = mode["builder"]()
        user_prompt = template.build()
        user_prompt += f"\n\n## Data to Analyze\n{test_data}"

        try:
            reply = call_llm(SYSTEM_PROMPT, user_prompt)
            print(reply)
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
