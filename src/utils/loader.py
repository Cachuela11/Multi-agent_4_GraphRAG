"""
文件加载与语义化切分 —— 支持 PDF / Markdown
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 默认切分参数（可通过 .env 覆盖）
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def _get_splitter() -> RecursiveCharacterTextSplitter:
    chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def load_pdf(file_path: str | Path) -> list[dict]:
    """加载 PDF 文件，返���切分后的文档片段列表"""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    splitter = _get_splitter()
    chunks = splitter.split_documents(pages)
    return [
        {
            "content": chunk.page_content,
            "source_file": os.path.basename(str(file_path)),
            "page": chunk.metadata.get("page", 0),
        }
        for chunk in chunks
    ]


def load_markdown(file_path: str | Path) -> list[dict]:
    """加载 Markdown 文件，返回切分后的文档片段列表"""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    splitter = _get_splitter()
    chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "source_file": path.name,
            "page": None,
        }
        for chunk in chunks
    ]


def load_directory(dir_path: str | Path) -> list[dict]:
    """遍历目录，加载所有 PDF 和 Markdown 文件"""
    dir_path = Path(dir_path)
    all_docs: list[dict] = []

    for file in dir_path.rglob("*"):
        if file.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(file))
        elif file.suffix.lower() in (".md", ".markdown"):
            all_docs.extend(load_markdown(file))

    print(f"[Loader] 共加载 {len(all_docs)} 个文档片段")
    return all_docs
