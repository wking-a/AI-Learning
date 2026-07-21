# Day 1：LLM API 基础 & 第一个 AI 应用

**日期**：2026-07-08

---

## 1. 核心概念

### 1.1 LLM API 的本质

LLM（大语言模型）本质上是一个**文本生成引擎**。你给它一段文本（Prompt），它返回续写的文本（Completion）。

```
输入： "中国的首都是"
输出： "北京。中国是一个拥有悠久历史的国家..."
```

**关键理解：这不是传统程序**——同样的输入可能返回不同的输出，因为 LLM 本质是"下一个词预测器"，每次从概率分布中采样。

### 1.2 Chat Completion API 的消息结构

不管 OpenAI、DeepSeek 还是其他 LLM API，都遵循同样的消息结构：

```python
messages = [
    {"role": "system",    "content": "你是一个 AI 助手"},    # 系统设定
    {"role": "user",      "content": "你好"},                # 用户输入
    {"role": "assistant", "content": "你好！有什么可以帮你的"}, # 模型回复
    {"role": "user",      "content": "北京有多大"}            # 继续对话
]
```

**为什么一定是 List？**
因为 List 是**有序序列**。Dict 无序，无法表达"谁先谁后"。LLM 依赖顺序来理解时间线和上下文关系。把 messages 传给模型，等同于给了它一段连续文本：

```
[system] 你是一个 AI 助手
[user] 你好
[assistant] 你好！
[user] 北京有多大
→ 模型预测下一个词是什么
```

### 1.3 三个关键参数

| 参数 | 作用 | 范围 | 推荐值 |
|------|------|------|--------|
| `temperature` | 控制概率分布的"尖锐程度" | 0~2 | 客服 0，创意 0.8 |
| `max_tokens` | 最大输出长度 | 依模型而定 | 按需设置 |
| `model` | 模型选择 | 各平台不同 | 按成本/能力权衡 |

**Temperature 的原理：**
- 0：总是选最高概率的词，输出确定
- 0.8~1：概率分布自然，平衡
- >1：概率分布平坦化，低概率词也可能被选中

**注意：Temperature=0 时输出仍然不完全确定。** 原因是浮点数精度问题 + 采样实现差异，另外 Top-P、Frequency Penalty 等参数也在交互影响。

### 1.4 API 无状态设计

**这是最重要的一个理解。**

每次 API 调用都是独立的。模型不会记住上一次调用。这意味着：
- 你必须自己把整个对话历史发过去
- 对话越长，Token 消耗越大
- 你需要自己实现"记忆"功能

**为什么这样设计？**
这是**设计取舍**：牺牲便利性换取可扩展性。无状态让 API Server 可以水平扩展（不需要共享内存）。如果 API Server 需要维护状态，每次请求都要路由到同一台机器，无法弹性扩缩。

### 1.5 API 返回结构的设计意图

```python
response.choices[0].message.content
```

嵌套结构的含义：

| 层级 | 为什么存在 |
|------|-----------|
| `choices` | 支持 `n>1` 返回多个候选（选最好的） |
| `message` | 格式和请求的 message 一致，可直接放回 messages 列表 |
| `content` | 区分普通文本和将来的 `tool_calls`、`function_call` |

设计思想：API 的返回结构就是为**继续对话**设计的——直接把 `response.choices[0].message` append 回 `messages` 就是下一轮对话。

---

## 2. 今日 Demo

### 2.1 项目结构

```
demos/day01/
├── chat.py              # 交互式 Chat CLI
├── requirements.txt     # openai + python-dotenv
├── .env                 # API Key（不要提交 Git！）
└── .env.example         # 模板
```

### 2.2 chat.py 核心设计

```python
def chat(messages: list, temperature: float = 0.7) -> str:
    """调用 LLM API 获取回复"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
```

**关键工程细节：**
- `load_dotenv()` 不带路径时存在隐患——从其他目录运行会加载失败
- 错误处理只捕了 `Exception`，没有区分 API 错误（401/429）和网络错误
- messages 列表由客户端维护，每次循环手动 append user 和 assistant 消息

### 2.3 运行

```bash
cd demos/day01
pip install -r requirements.txt
# 配置 .env 文件
python chat.py
```

API 提供商选择（本项目使用 DeepSeek）：
- DeepSeek：`LLM_BASE_URL=https://api.deepseek.com`，模型 `deepseek-chat`
- OpenAI：`LLM_BASE_URL=https://api.openai.com/v1`，模型 `gpt-4o-mini`

---

## 3. 架构设计

```
┌─────────────┐
│    User     │  (终端输入)
└──────┬──────┘
       │ "你好"
       ▼
┌─────────────┐
│   chat.py   │ ← 维护 messages 列表
│  CLI 客户端   │ ← 管理对话历史
└──────┬──────┘
       │ POST /v1/chat/completions
       ▼
┌─────────────┐
│  DeepSeek   │ ← 无状态推理
│  API Server │ ← 每次调用独立
└──────┬──────┘
       │ response JSON
       ▼
┌─────────────┐
│  messages   │ ← append assistant reply
│  列表更新    │ → 进入下一轮循环
└─────────────┘
```

**关键数据流：**
1. User 输入 → 追加到 messages
2. 完整 messages → API
3. API 回复 → 追加到 messages
4. 循环回到 1

**设计思想：** 这是最简单的"无状态 API + 客户端维护上下文"架构。所有 AI 应用的基础模式。

---

## 4. 面试重点（10题）

### Q1：什么是 Temperature？它的作用？
Temperature 控制输出概率分布的"尖锐程度"。值越低越确定，越高越多样。范围 0~2。
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
让 API Server 可以**水平扩展**（不需要共享内存）。每一次 API 请求都是独立的，服务器默认不会记住你之前说过什么。这是**权衡**：牺牲便利性换取可扩展性。

### Q9：API 返回内容被截断可能的原因？
1. `max_tokens` 设置过短
2. 超过模型上下文窗口限制
3. 触发了 Content Filter

### Q10：如何实现"记忆"功能？
把 Messages 存到数据库（Redis/SQLite），下次对话时加载历史 messages 传入 API。

---

## 5. 开放思考题

1. **为什么 OpenAI 的 API 不帮我们维护 History？** 如果 OpenAI 帮你存 History，你会有什么顾虑？（隐私？合规？跨服务绑定？）
2. **Chat Completion API 为什么用 POST 而不是 GET？** GET 也能传数据，为什么行业标准用 POST？（请求体大小？安全性？幂等性？）
3. **如果有一天 LLM API 彻底免费了，现在的架构会发生什么变化？** 哪些设计会变得不再重要？（Token 优化？Prompt 压缩？）
4. **为什么请求体是 JSON 而不是 Protobuf 或 XML？** 在延迟敏感的 AI 场景下，JSON 解析的开销为什么被接受了？（生态成熟度？调试便利性？）

---

## 6. Risk Copilot 项目

**版本**：v0.1

**文件**：`projects/risk_copilot/main.py`

**功能**：风控领域 AI 问答助手

**关键设计**：
- 使用专业风控 System Prompt，覆盖反欺诈、信用评分、交易风控等领域
- Temperature=0.3，保证回答的确定性和专业性
- 硬编码了 `.env` 绝对路径（工程上不灵活，Day2 改进）

**项目理念**：整个 30 天训练营只维护这一个项目——Risk Copilot。每天增加一个能力，从今天的"聊天"一路演进到 Day30 的完整系统。

**工程问题记录**（后面优化）：
1. `.env` 路径硬编码
2. 错误处理粒度不够
3. 没有 Prompt 模板系统

---

## 7. 今日总结

### 真正掌握的能力
- ✅ 理解 Chat Completion API 的消息结构
- ✅ 理解 API 为什么设计为无状态
- ✅ 掌握 Temperature、Max Tokens 等核心参数
- ✅ 能用 Python 调用 LLM API
- ✅ 能自己维护 Messages 列表管理上下文
- ✅ 理解 API 返回结构的嵌套设计意图
- ✅ 启动 Risk Copilot 项目

### 还不会的（明天解决）
- ❌ Token 的精确计算方式
- ❌ 如何处理超长对话
- ❌ Streaming 模式

### 🎯 Mentor 一句话
> Chat Completion API 的本质是一个"无状态的文本生成引擎"，Messages List 就是上下文——开发者自己维护它，而不是模型记住它。

### 明日预告
Day 2：Prompt Engineering 系统方法论——Chain-of-Thought、Few-shot、结构化 Prompt。
