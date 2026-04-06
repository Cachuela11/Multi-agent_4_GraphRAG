"""
Planner 节点 —— 接收用户问题，拆解为可执行的子步骤
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils.memory import load_long_term_memory


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的规划器。\n"
        "你的任务是将用户的问题拆解为 2-5 个可独立检索并回答的子步骤。\n"
        "每个子步骤应该是一个具体的信息检索或推理任务。\n"
        "请以 JSON 数组格式输出步骤列表，每个元素是一个字符串。\n\n"
        "历史知识（长期记忆）：\n{long_term_memory}\n",
    ),
    ("human", "{query}"),
])


def planner_node(state: dict) -> dict:
    """
    Planner 节点逻辑：
    1. 读取长期记忆作为背景知识
    2. 调用 LLM 将 query 拆解为子步骤
    3. 返回 plan 列表并初始化执行状态
    """
    import json

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    long_mem = load_long_term_memory()
    long_mem_str = json.dumps(long_mem, ensure_ascii=False) if long_mem else "无"

    chain = PLANNER_PROMPT | llm
    result = chain.invoke({
        "query": state["query"],
        "long_term_memory": long_mem_str,
    })

    # 解析 LLM 输出为步骤列表
    try:
        plan = json.loads(result.content)
        if not isinstance(plan, list):
            plan = [result.content]
    except json.JSONDecodeError:
        plan = [result.content]

    return {
        "plan": plan,
        "current_step": 0,
        "steps_results": [],
        "sources": [],
        "short_term_memory": [f"用户问题: {state['query']}"],
        "needs_revision": False,
    }
