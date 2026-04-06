"""
混合检索与重排序 —— Dense (ChromaDB) + BM25 + RRF + Cross-Encoder
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import chromadb

import jieba
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.utils.loader import load_directory


# ── 模块级单例 ──────────────────────────────────────────────

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_embeddings: HuggingFaceEmbeddings | None = None

_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None

_reranker: CrossEncoder | None = None


# ── ChromaDB / Embedding 初始化 ────────────────────────────

def _get_collection() -> chromadb.Collection:
    """获取或初始化 ChromaDB 集合（单例）"""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./index")
    _chroma_client = chromadb.PersistentClient(path=persist_dir)
    _collection = _chroma_client.get_or_create_collection(
        name="campus_docs",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _get_embeddings() -> HuggingFaceEmbeddings:
    """获取 HuggingFace 本地 Embedding 模型（单例，首次调用自动下载）"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[Retriever] Embedding 模型已加载: {model_name}")
    return _embeddings


# ── Cross-Encoder Reranker ─────────────────────────────────

def _get_reranker() -> CrossEncoder:
    """获取 Cross-Encoder 重排序模型（单例，首次调用自动下载）"""
    global _reranker
    if _reranker is not None:
        return _reranker

    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    _reranker = CrossEncoder(model_name)
    print(f"[Retriever] Reranker 模型已加载: {model_name}")
    return _reranker


# ── BM25 工具函数 ──────────────────────────────────────────

def _tokenize_chinese(text: str) -> list[str]:
    """使用 jieba 对中文文本分词，用于 BM25 索引与查询"""
    return list(jieba.cut(text))


def _build_bm25_index(texts: list[str], corpus_meta: list[dict]) -> BM25Okapi:
    """
    构建 BM25 索引并持久化到磁盘。

    Args:
        texts: 文档内容列表
        corpus_meta: 对应的元数据列表 [{"content", "source_file", "page"}, ...]
    """
    global _bm25_index, _bm25_corpus

    tokenized = [_tokenize_chinese(t) for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = corpus_meta

    bm25_path = os.getenv("BM25_INDEX_PATH", "./index/bm25_index.pkl")
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": _bm25_index, "corpus": _bm25_corpus}, f)

    print(f"[Retriever] BM25 索引已构建: {len(texts)} 个片段")
    return _bm25_index


def _load_bm25_index() -> BM25Okapi | None:
    """
    加载 BM25 索引。
    优先级: 内存单例 > 磁盘 pickle > 从 ChromaDB 重建。
    """
    global _bm25_index, _bm25_corpus

    if _bm25_index is not None:
        return _bm25_index

    bm25_path = os.getenv("BM25_INDEX_PATH", "./index/bm25_index.pkl")

    # 尝试从 pickle 加载
    if Path(bm25_path).exists():
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            _bm25_index = data["bm25"]
            _bm25_corpus = data["corpus"]
            print(f"[Retriever] BM25 索引已从磁盘加载: {len(_bm25_corpus)} 个片段")
            return _bm25_index
        except Exception as e:
            print(f"[Retriever] BM25 pickle 加载失败，将从 ChromaDB 重建: {e}")

    # 从 ChromaDB 重建
    collection = _get_collection()
    if collection.count() == 0:
        return None

    all_data = collection.get(include=["documents", "metadatas"])
    texts = all_data["documents"]
    corpus_meta = [
        {
            "content": texts[i],
            "source_file": all_data["metadatas"][i].get("source_file", "unknown"),
            "page": all_data["metadatas"][i].get("page"),
        }
        for i in range(len(texts))
    ]

    return _build_bm25_index(texts, corpus_meta)


# ── 双路检索 ──────────────────────────────────────────────

def _dense_search(query: str, top_k: int) -> list[dict]:
    """稠密向量检索（ChromaDB cosine）"""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embeddings = _get_embeddings()
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "content": results["documents"][0][i],
            "source_file": results["metadatas"][0][i].get("source_file", "unknown"),
            "page": results["metadatas"][0][i].get("page"),
            "relevance_score": 1 - results["distances"][0][i],
        })
    return docs


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """BM25 稀疏检索（适合缩写/关键词精确匹配）"""
    bm25 = _load_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = _tokenize_chinese(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        doc = _bm25_corpus[idx]
        results.append({
            "content": doc["content"],
            "source_file": doc["source_file"],
            "page": doc["page"],
            "relevance_score": float(scores[idx]),
        })
    return results


# ── RRF 融合 ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion: 合并多路检索结果。
    RRF(d) = Σ 1 / (k + rank(d))  对每一路结果求和。
    以文档 content 去重。
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc["content"]
            if key not in doc_map:
                doc_map[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    sorted_keys = sorted(fused_scores, key=lambda c: fused_scores[c], reverse=True)

    return [
        {**doc_map[key], "relevance_score": fused_scores[key]}
        for key in sorted_keys
    ]


# ── Cross-Encoder 重排序 ──────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """使用 Cross-Encoder 对候选文档重排序，返回 top_k 个最相关文档"""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc["content"]) for doc in candidates]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(candidates, scores), key=lambda x: x[1], reverse=True
    )

    return [
        {**doc, "relevance_score": float(score)}
        for doc, score in scored_docs[:top_k]
    ]


# ── 公开接口 ──────────────────────────────────────────────

def get_index_count() -> int:
    """返回当前索引中的文档片段数量"""
    return _get_collection().count()


def index_documents(data_dir: str = "./data", force: bool = False) -> int:
    """
    从 data_dir 加载文档并建立向量索引 + BM25 索引。
    如果索引已存在且 force=False，则跳过。
    返回索引中的文档片段数量。
    """
    collection = _get_collection()
    existing = collection.count()

    if existing > 0 and not force:
        print(f"[Retriever] 索引已存在 ({existing} 个片段)，跳过重建。使用 --reindex 强制重建。")
        return existing

    docs = load_directory(data_dir)
    if not docs:
        print("[Retriever] data 目录为空，跳过索引")
        return 0

    # 强制重建时先清空旧数据
    if force and existing > 0:
        _chroma_client.delete_collection("campus_docs")
        globals()["_collection"] = None
        collection = _get_collection()
        print("[Retriever] 已清空旧索引")

    embeddings = _get_embeddings()

    texts = [d["content"] for d in docs]
    metadatas = [
        {"source_file": d["source_file"], "page": d.get("page") or 0}
        for d in docs
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]

    # 批量嵌入
    vectors = embeddings.embed_documents(texts)

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[Retriever] 已索引 {len(docs)} 个文档片段")

    # 构建 BM25 索引
    corpus_meta = [
        {"content": d["content"], "source_file": d["source_file"], "page": d.get("page")}
        for d in docs
    ]
    _build_bm25_index(texts, corpus_meta)

    return len(docs)


def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """
    混合检索: Dense + BM25 → RRF 融合 → Cross-Encoder 重排序。
    返回 top_k 个最相关文档片段。

    返回格式: [{"content", "source_file", "page", "relevance_score"}, ...]
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    rrf_k = int(os.getenv("RRF_K", "60"))
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "20"))

    # 1. 双路检索
    dense_results = _dense_search(query, top_k=reranker_top_k)
    bm25_results = _bm25_search(query, top_k=reranker_top_k)

    # 2. RRF 融合
    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)

    # 3. Cross-Encoder 重排序
    candidates = fused[:reranker_top_k]
    reranked = _rerank(query, candidates, top_k=top_k)

    return reranked
