import argparse
import os
from typing import List

from config import get_es
from es_functions import create_elastic_index
from embedding import local_embedding
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieve_documents import elastic_search, rerank


def ensure_index(index_name: str):
    es = get_es()
    if not es.indices.exists(index=index_name):
        create_elastic_index(index_name)


def ingest_pdf(index_name: str, pdf_path: str, batch_size: int = 25, use_local: bool = False):
    """最小文本入库流程：加载→分块→批量向量化→写入 ES（仅 text+vector）。
    
    参数：
    - use_local: 是否使用本地模型（True=本地，False=远程服务，默认False）
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    es = get_es()
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    batch_docs: List = []  # 保留 Document 对象以获取元数据
    for i, chunk in enumerate(chunks):
        batch_docs.append(chunk)
        is_last = (i == len(chunks) - 1)

        if len(batch_docs) == batch_size or is_last:
            # 提取文本用于向量化
            texts = [d.page_content for d in batch_docs]
            # 根据用户选择使用本地或远程向量服务
            vectors = local_embedding(texts, use_local=use_local)
            
            # 写入 ES（附带元数据）
            for doc, text, vec in zip(batch_docs, texts, vectors):
                md = getattr(doc, "metadata", {}) or {}
                # 提取文件名（通用方法）
                src = md.get("source") or pdf_path
                file_name = os.path.basename(src)
                # 提取页码（PyMuPDFLoader 提供 page，0-based）
                page0 = md.get("page")
                
                body = {"text": text, "vector": vec, "file_name": file_name, "doc_type": "text"}
                if isinstance(page0, int):
                    body["page"] = page0 + 1  # 转为 1-based 人类友好页码
                
                es.index(index=index_name, body=body)
            batch_docs = []


def query_index(index_name: str, query: str, top_k: int = 5, use_rerank: bool = False, use_local: bool = False):
    """查询索引并返回结果。
    
    参数：
    - use_local: 是否使用本地模型（True=本地，False=远程服务，默认False）
    """
    results = elastic_search(query, index_name, use_local=use_local)
    
    # 调试：显示 rerank 前的排序
    if use_rerank:
        print("\n--- Before Rerank (RRF scores) ---")
        for r in results[:5]:
            preview = (r.get("text") or "")[:80].replace("\n", " ")
            file_name = r.get("file_name", "未知文件")
            page = r.get("page")
            score = r.get("score", 0)
            print(f"  [{file_name} - p.{page}] RRF_score={score:.6f}: {preview}")

    if use_rerank:
        # 先取较多候选做重排，再展示前 top_k
        candidate_k = max(20, top_k)
        results = rerank(query, results[:candidate_k])
        
        # 调试：显示 rerank 后的排序
        print("\n--- After Rerank (Reranker scores) ---")
        for r in results[:5]:
            preview = (r.get("text") or "")[:80].replace("\n", " ")
            file_name = r.get("file_name", "未知文件")
            page = r.get("page")
            score = r.get("score", 0)
            print(f"  [{file_name} - p.{page}] rerank_score={score:.6f}: {preview}")
    
    # 显示检索结果（带引用）
    for r in results[:top_k]:
        preview = (r.get("text") or "")[:160].replace("\n", " ")
        file_name = r.get("file_name", "未知文件")
        page = r.get("page")
        ref = f"[{file_name}"
        if page:
            ref += f" - p.{page}"
        ref += "]"
        print(f"#{r['rank']}: {preview} {ref}")

    # 生成带引用的答案（展示 rank 与 score）
    answer_context = "\n".join((res.get("text") or "") for res in results[:top_k])
    references = []
    seen_refs = set()
    for idx, res in enumerate(results[:top_k], start=1):
        file_name = res.get("file_name", "未知文件")
        page = res.get("page")
        score = res.get("score")
        ref_key = f"{file_name}_{page}"
        if ref_key not in seen_refs:
            seen_refs.add(ref_key)
            ref = f"[{file_name}"
            if page:
                ref += f" - p.{page}"
            ref += "]"
            if score is not None:
                ref += f"  | score={score:.4f}"
            if idx == 1:
                ref += "  | TOP1"
            references.append(ref)
    
    print("\n--- Draft Answer (Context-based) ---\n")
    print(answer_context[:1200])
    if references:
        print("\n--- References (rank/score) ---")
        for ref in references:
            print(ref)


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG text pipeline")
    parser.add_argument("--index", required=True, help="Elasticsearch index name")
    parser.add_argument("--pdf", help="PDF file to ingest (text only)")
    parser.add_argument("--query", help="Query string to search")
    parser.add_argument("--top_k", type=int, default=5, help="Top K to print")
    parser.add_argument("--rerank", action="store_true", help="Apply reranker (may cause instability - use with caution)")
    parser.add_argument("--use-local", action="store_true", help="Use local embedding model (default: remote service)")
    args = parser.parse_args()

    ensure_index(args.index)

    if args.pdf:
        mode = "local" if args.use_local else "remote"
        print(f"[Ingest] {args.pdf} -> {args.index} (embedding: {mode})")
        ingest_pdf(args.index, args.pdf, use_local=args.use_local)

    if args.query:
        mode = "local" if args.use_local else "remote"
        print(f"[Search] index={args.index} query=\"{args.query}\" rerank={args.rerank} (embedding: {mode})")
        query_index(args.index, args.query, top_k=args.top_k, use_rerank=args.rerank, use_local=args.use_local)


if __name__ == "__main__":
    main()


