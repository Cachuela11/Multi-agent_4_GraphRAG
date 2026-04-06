"""
Reporter 节点 —— 汇总所有结果生成最终结构化回答
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils.memory import save_long_term_memory


REPORTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的报告生成器。\n"
        "你需要将各子步骤的执行结果汇总为一份结构化回答。\n"
        "要求：\n"
        "- 直接回答用户问题\n"
        "- 引用关键证据来源\n"
        "- 如果有不确定的部分，明确说明\n"
        "- 使用清晰的 Markdown 格式\n",
    ),
    (
        "human",
        "用户问题: {query}\n\n"
        "各步骤结果:\n{steps_results}\n\n"
        "证据来源:\n{sources}\n\n"
        "请生成最终回答。",
    ),
])


def reporter_node(state: dict) -> dict:
    """
    Reporter 节点逻辑：
    1. 汇总 steps_results 和 sources
    2. 调用 LLM 生成结构化回答
    3. 将问答对写入长期记忆
    """
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.3,
    )

    steps_text = "\n\n".join(
        f"步骤 {i+1}: {r['step']}\n结果: {r['result']}"
        for i, r in enumerate(state["steps_results"])
    )

    # 去重证据来源
    seen = set()
    unique_sources = []
    for s in state["sources"]:
        key = (s["source_file"], s.get("page"))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    sources_text = "\n".join(
        f"- [{s['source_file']}](p.{s.get('page', '?')}) "
        f"相关度: {s.get('relevance_score', 0):.2f}"
        for s in unique_sources[:10]
    )

    chain = REPORTER_PROMPT | llm
    result = chain.invoke({
        "query": state["query"],
        "steps_results": steps_text,
        "sources": sources_text or "无来源",
    })

    # 写入长期记忆
    save_long_term_memory(state["query"], result.content)

    return {
        "response": result.content,
        "short_term_memory": ["最终回答已生成"],
    }
