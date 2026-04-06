"""
双层记忆系统 —— 短期记忆 + 长期记忆
"""

from __future__ import annotations

import json
from pathlib import Path

# 长期记忆持久化文件
MEMORY_FILE = Path("./index/long_term_memory.json")


# ---------------------------------------------------------------------------
# 长期记忆（跨会话持久化）
# ---------------------------------------------------------------------------

def load_long_term_memory() -> dict:
    """加载长期记忆，返回 {query: answer} 形式的字典"""
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_long_term_memory(query: str, answer: str) -> None:
    """将问答对追加到长期记忆"""
    memory = load_long_term_memory()

    # 保留最近 100 条记录，防止无限增长
    if len(memory) >= 100:
        keys = list(memory.keys())
        for k in keys[: len(keys) - 99]:
            del memory[k]

    memory[query] = answer

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 短期记忆（会话级，由 State 管理，此处提供辅助函数）
# ---------------------------------------------------------------------------

def summarize_short_term(messages: list[str], max_items: int = 20) -> list[str]:
    """保留最近 max_items 条短期记忆"""
    return messages[-max_items:]
