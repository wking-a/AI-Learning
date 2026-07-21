"""
Day 2: Prompt Builder
=====================
结构化 Prompt 构建器 — 把 Prompt 当作代码组织

功能：
  - 模板化管理 Prompt 各部分
  - 支持动态变量填充
  - 多种策略模式 (Direct / CoT / Few-shot)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptTemplate:
    """
    结构化 Prompt 模板

    将 Prompt 拆解为独立组件，最终组装成完整 Prompt。
    """

    role: str = ""
    task: str = ""
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    output_format: str = ""
    examples: list[tuple[str, str]] = field(default_factory=list)

    def build(self, **kwargs) -> str:
        """
        组装最终 Prompt

        参数:
            **kwargs: 动态填充变量，如 {"question": "..."}

        返回:
            完整的 Prompt 文本
        """
        parts = []

        if self.role:
            parts.append(f"## Role\n{self.role}")

        if self.task:
            # 支持动态变量填充 {var_name}
            task = self.task.format(**kwargs) if kwargs else self.task
            parts.append(f"## Task\n{task}")

        if self.context:
            ctx = self.context.format(**kwargs) if kwargs else self.context
            parts.append(f"## Context\n{ctx}")

        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"## Constraints\n{constraints_str}")

        if self.output_format:
            parts.append(f"## Output Format\n{self.output_format}")

        if self.examples:
            examples_str = "\n".join(
                f"输入: {inp}\n输出: {out}" for inp, out in self.examples
            )
            parts.append(f"## Examples\n{examples_str}")

        return "\n\n".join(parts)


# ── 预置模板 ──────────────────────────────────

def risk_analysis_template() -> PromptTemplate:
    """风控分析模板"""
    return PromptTemplate(
        role="资深风控专家，精通反欺诈、信用评分、交易风控。",
        task="分析以下交易/行为是否存在风险，给出风险评估结果。",
        constraints=[
            "严格基于给定数据分析，不要猜测",
            "如果信息不足，明确指出缺少什么信息",
            "给出风险等级：低 / 中 / 高 / 极高",
            "每种风险等级必须说明理由",
        ],
        output_format="""风险等级: [等级]
理由: [详细分析]
建议: [处置建议]
缺少信息: [如果有的話]""",
    )


def chain_of_thought_template() -> PromptTemplate:
    """思维链模板"""
    return PromptTemplate(
        role="你是一个逻辑严密的推理专家，擅长逐步分析问题。",
        task="请逐步推理并回答以下问题：{question}",
        constraints=[
            "先列出所有已知信息",
            "一步一步推理，每一步都要有依据",
            "最后给出结论",
        ],
        output_format="""已知信息:
- ...

推理过程:
Step 1: ...
Step 2: ...
...

结论:
...""",
    )


def extractor_template() -> PromptTemplate:
    """信息抽取模板（Few-shot）"""
    return PromptTemplate(
        role="精确的信息抽取器。",
        task="从以下文本中抽取指定字段。",
        constraints=[
            "只抽取文本中明确出现的信息",
            "如果字段不存在，输出 null",
            "不要编造信息",
        ],
        output_format="""{
  "person_name": "... 或 null",
  "date": "... 或 null",
  "amount": "... 或 null",
  "risk_type": "... 或 null"
}""",
        examples=[
            (
                "客户张伟于2024年3月15日投诉账户被盗，损失金额5000元。",
                '{"person_name": "张伟", "date": "2024-03-15", "amount": "5000", "risk_type": "盗刷"}',
            ),
            (
                "今天收到一条短信说我的银行卡异常。",
                '{"person_name": null, "date": null, "amount": null, "risk_type": "钓鱼"}',
            ),
        ],
    )
