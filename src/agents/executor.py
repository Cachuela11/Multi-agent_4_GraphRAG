"""
Executor 节点 —— 逐步执行子任务，包含 DeepResearch 探索链
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils.retriever import retrieve_documents


EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的执行器。\n"
        "你需要根据给定的子任务和检索到的相关文档来回答问题。\n"
        "请基于证据进行推理，如果文档不足以回答，请明确说明。\n\n"
        "当前上下文（短期记忆）：\n{short_term_memory}\n\n"
        "检索到的文档片段：\n{retrieved_docs}\n",
    ),
    ("human", "子任务: {step}\n\n请给出回答，并标注你引用了哪些文档。"),
])

# DeepResearch：最多进行 N 轮迭代检索
MAX_DEEP_RESEARCH_ROUNDS = 3


def _deep_research(step: str, initial_docs: list[dict], llm: ChatOllama) -> dict:
    """
    DeepResearch 探索链：
    1. 第一轮用原始 step 检索
    2. 根据结果判断是否需要追加检索（证据不充分时生成新 query）
    3. 最多 MAX_DEEP_RESEARCH_ROUNDS 轮，合并所有证据
    """
    all_docs = list(initial_docs)
    evidence_chain = []

    for round_idx in range(MAX_DEEP_RESEARCH_ROUNDS):
        docs_text = "\n---\n".join(
            f"[{d['source_file']}] {d['content']}" for d in all_docs
        ) or "无相关文档"

        result = EXECUTOR_PROMPT.format_messages(
            short_term_memory="",
            retrieved_docs=docs_text,
            step=step,
        )
        response = llm.invoke(result)

        evidence_chain.append({
            "round": round_idx + 1,
            "answer_snippet": response.content[:200],
            "num_docs": len(all_docs),
        })

        # 判断是否需要继续深入检索
        if "证据不足" not in response.content and "无法确定" not in response.content:
            break

        # 生成追加检索 query
        followup_docs = retrieve_documents(
            f"补充: {step} - {response.content[:100]}", top_k=3
        )
        all_docs.extend(followup_docs)

    return {
        "result": response.content,
        "sources": all_docs,
        "evidence_chain": evidence_chain,
    }


def executor_node(state: dict) -> dict:
    """
    Executor 节点逻辑：
    1. 取出当前步骤
    2. 向量检索相关文档
    3. DeepResearch 探索链
    4. 记录结果并推进 current_step
    """
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    plan = state["plan"]
    idx = state["current_step"]
    step = plan[idx]

    # 初始检索
    docs = retrieve_documents(step, top_k=5)

    # DeepResearch
    research = _deep_research(step, docs, llm)

    # 更新状态
    step_result = {
        "step": step,
        "result": research["result"],
        "evidence_chain": research["evidence_chain"],
    }

    new_sources = [
        {
            "content": d["content"],
            "source_file": d["source_file"],
            "page": d.get("page"),
            "relevance_score": d.get("relevance_score", 0.0),
        }
        for d in research["sources"]
    ]

    return {
        "current_step": idx + 1,
        "steps_results": state["steps_results"] + [step_result],
        "sources": state["sources"] + new_sources,
        "short_term_memory": [f"步骤 {idx + 1} 完成: {step}"],
    }
