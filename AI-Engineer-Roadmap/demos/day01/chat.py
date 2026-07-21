"""
Day 1: LLM Chat Demo
====================
功能：通过 LLM API 完成对话
目标：理解 Chat Completion API 的消息结构和无状态特性
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def chat(messages: list, temperature: float = 0.7) -> str:
    """
    调用 LLM API 获取回复

    参数:
        messages: 对话消息列表
            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."},
             {"role": "assistant", "content": "..."}]
        temperature: 随机性控制 (0-2)

    返回:
        模型回复文本
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def main():
    """简单的交互式 Chat CLI"""

    # 系统 Prompt：定义 AI 的行为
    system_prompt = "你是一个专业的 AI 助手。回答简洁、准确、有条理。"

    # 初始化消息列表
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    print("=" * 50)
    print("🤖 AI Chat Demo (Day 1)")
    print(f"模型: {MODEL}")
    print("输入 'exit' 退出，'clear' 清空历史")
    print("=" * 50)

    while True:
        # 用户输入
        user_input = input("\n👤 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("🧹 对话历史已清空")
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 调用 API
        print("🤖 AI: ", end="", flush=True)
        try:
            reply = chat(messages)
            print(reply)

            # 将回复加入历史（关键！API 无状态，需要自己维护上下文）
            messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            print(f"\n❌ 错误: {e}")

    print("\n👋 再见！")


if __name__ == "__main__":
    main()
