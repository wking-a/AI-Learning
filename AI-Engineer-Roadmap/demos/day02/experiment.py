"""
Day 2: Prompt Engineering 对比实验
==================================
同一个风控问题，用不同的 Prompt 策略，观察效果差异。

对比策略：
  1. 简单问答（无策略）
  2. 结构化 Prompt（Persona + Task + Constraints）
  3. Chain-of-Thought（思维链）
  4. Few-shot（示例学习）
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# 确保可以 import prompt_builder
sys.path.insert(0, os.path.dirname(__file__))
from prompt_builder import (
    PromptTemplate,
    risk_analysis_template,
    chain_of_thought_template,
    extractor_template,
)

load_dotenv(r"d:\AI-Learning\AI-Engineer-Roadmap\demos\day01\.env")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """统一的 LLM 调用"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── 测试数据 ──────────────────────────────────────
TEST_CASE = {
    "transaction": "用户王强在凌晨 2:30 使用新设备登录，尝试转账 50000 元到新添加的收款账户。该账户 30 分钟前刚刚注册。",
    "email": "尊敬的客户，您的账户存在异常活动，请点击以下链接验证信息：http://phishing.example.com",
    "extract": "昨日下午3点，我接到自称京东客服的电话，说我有一笔贷款需要处理，让我转账5000元到安全账户。",
}


# ── 策略 1：简单问答（基线） ─────────────────────
def strategy_naive():
    """没有任何策略，直接问"""
    print("\n" + "=" * 60)
    print("[Strategy 1] 简单问答（基线）")
    print("=" * 60)

    system = "你是一个 AI 助手。"
    user = f"分析这个交易是否有风险：{TEST_CASE['transaction']}"

    reply = call_llm(system, user)
    print(f"回复:\n{reply}\n")
    return reply


# ── 策略 2：结构化 Prompt ──────────────────────
def strategy_structured():
    """使用 PromptTemplate 构建的结构化 Prompt"""
    print("\n" + "=" * 60)
    print("[Strategy 2] 结构化 Prompt（Persona + Task + Constraints）")
    print("=" * 60)

    template = risk_analysis_template()
    system_prompt = "你是一个严格按照要求执行任务的 AI 助手。"

    user_prompt = template.build()
    # 追加实际要分析的数据
    user_prompt += f"\n\n## Data to Analyze\n{TEST_CASE['transaction']}"

    print(f"[Prompt 结构]:\n{user_prompt}\n")
    reply = call_llm(system_prompt, user_prompt)
    print(f"回复:\n{reply}\n")
    return reply


# ── 策略 3：Chain-of-Thought ─────────────────────
def strategy_cot():
    """思维链 Prompt"""
    print("\n" + "=" * 60)
    print("[Strategy 3] Chain-of-Thought（思维链）")
    print("=" * 60)

    template = chain_of_thought_template()
    system_prompt = "你是一个严格的逐步推理专家。"

    # 动态填充 question
    question = f"分析这比交易的风险等级（低/中/高/极高），并给出理由：{TEST_CASE['transaction']}"
    user_prompt = template.build(question=question)

    print(f"[Prompt]:\n{user_prompt}\n")
    reply = call_llm(system_prompt, user_prompt)
    print(f"回复:\n{reply}\n")
    return reply


# ── 策略 4：Few-shot ────────────────────────────
def strategy_fewshot():
    """带示例的 Few-shot Prompt"""
    print("\n" + "=" * 60)
    print("[Strategy 4] Few-shot（示例学习）")
    print("=" * 60)

    template = extractor_template()
    system_prompt = "你是一个精确的信息抽取器。"

    user_prompt = template.build()
    user_prompt += f"\n\n## Input Text\n{TEST_CASE['extract']}"

    print(f"[Prompt]:\n{user_prompt}\n")
    reply = call_llm(system_prompt, user_prompt)
    print(f"回复:\n{reply}\n")
    return reply


# ── 策略 5：完整结构化 + 多角度评估 ─────────────
def strategy_full():
    """完整结构化 Prompt，要求多角度评估"""
    print("\n" + "=" * 60)
    print("[Strategy 5] 完整结构化 + 多角度评估")
    print("=" * 60)

    system_prompt = "你是一名资深的金融风控专家。你严格遵循评估框架进行分析。"

    user_prompt = f"""## Role
资深金融风控专家

## Task
对以下交易进行多角度风险评估

## Risk Assessment Framework
请从以下维度逐一分析，每个维度给出评分 (1-5) 和理由:

### 1. 时间异常 (Time Anomaly)
- 交易时间是否在常规时段？
- 理由必须包含具体时间分析

### 2. 设备风险 (Device Risk)
- 是否为新设备？
- 与历史行为是否一致？

### 3. 金额风险 (Amount Risk)
- 金额大小是否异常？
- 是否接近阈值？

### 4. 收款方风险 (Recipient Risk)
- 收款方是否是首次交易？
- 收款方账户特征？

## Transaction Data
{TEST_CASE['transaction']}

## Output Format
**最终风险等级**: [低/中/高/极高]
**综合评分**: [总分/20]

### 分维度评分:
1. 时间异常: [1-5] - 理由
2. 设备风险: [1-5] - 理由
3. 金额风险: [1-5] - 理由
4. 收款方风险: [1-5] - 理由

**处置建议**: [具体建议]
"""

    print(f"[Prompt 长度]: {len(user_prompt)} 字符\n")
    reply = call_llm(system_prompt, user_prompt, temperature=0.2)
    print(f"回复:\n{reply}\n")
    return reply


# ── 主程序 ──────────────────────────────────────
def main():
    print("=" * 60)
    print("[Experiment] Prompt Engineering 策略对比实验")
    print(f"模型: {MODEL}  |  Temperature: 0.3")
    print("=" * 60)

    results = {}

    for name, func in [
        ("1_naive", strategy_naive),
        ("2_structured", strategy_structured),
        ("3_cot", strategy_cot),
        ("4_fewshot", strategy_fewshot),
        ("5_full", strategy_full),
    ]:
        try:
            reply = func()
            results[name] = reply
        except Exception as e:
            print(f"[ERROR] {e}")
            results[name] = f"[ERROR] {e}"

    # ── 对比总结 ──
    print("\n" + "=" * 60)
    print("[Summary] 策略对比总结")
    print("=" * 60)
    print(f"{'策略':<25} {'回复长度':<12} {'是否结构化':<12}")
    print("-" * 50)
    for name, reply in results.items():
        length = len(reply) if not reply.startswith("[ERROR]") else 0
        structured = "是" if name != "1_naive" else "否"
        print(f"{name:<25} {length:<12} {structured:<12}")

    print("\n[Observation] 观察结论:")
    print("1. 结构化 Prompt 回复更规范，更容易做后续处理")
    print("2. CoT 让模型展示推理过程，结果更可信")
    print("3. Few-shot 适合格式化的抽取任务")
    print("4. 多角度评估框架能得到更全面的分析")


if __name__ == "__main__":
    main()
