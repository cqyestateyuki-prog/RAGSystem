## 课件与代码模块映射指南

本指南帮助你将 `rag-presentation.html` 中的知识点与代码实现一一对应，方便学习与实验。

- 第 2 页：Why RAG（问题与动机）
  - 概念：无需代码，作为背景理解。

- 第 3 页：基础架构（Document Store / Embedding / Generator）
  - 代码：`es_functions.py`（索引）、`embedding.py`（向量）、`retrieve_documents.py`（检索）。

- 第 4 页：Embedding 模型与余弦相似度
  - 代码：`embedding.py`、`es_functions.py` 中 `dense_vector` + `similarity: cosine`；
  - 检索处的脚本评分：`retrieve_documents.py` 中 `script_score` + `cosineSimilarity`。

- 第 5 页：PDF 处理与分块（Chunking）
  - 代码：`document_process.py`（`RecursiveCharacterTextSplitter`，`chunk_size=1024`，`chunk_overlap=100`）。

- 第 6 页：Retrieve & Rerank（Hybrid + RRF + Neural Rerank）
  - 关键词（BM25）：`retrieve_documents.py` `keyword_query`（基于分词关键词）
  - 向量检索：`retrieve_documents.py` `vector_query`
  - RRF 融合：`retrieve_documents.py` `hybrid_search_rrf`
  - 神经重排：`retrieve_documents.py` `rerank`（需要 `RERANK_URL`）

- 第 7 页：RAG Fusion（多查询合并）
  - 代码：`retrieve_documents.py` `rag_fusion`
  - 用法：生成多个改写查询，分别检索后再融合（示例在 README）。

- 第 8 页：多轮对话的指代消解
  - 代码：`retrieve_documents.py` `coreference_resolution`

- 第 9 页：查询分解（Query Decomposition）
  - 代码：`retrieve_documents.py` `query_decompositon`

- 第 10 页：元数据过滤（Metadata Filtering）
  - 代码：`es_functions.py` 中注释的映射字段（`file_name/page/chapter/...`）可启用；
  - 思路：创建映射 -> 写入时附带元数据 -> 检索时在 `query/knn` 同时加 `filter`。

- 第 11 页：Web Search 作为 RAG
  - 代码：`websearch.py`（`bocha_web_search` 拉取网页，`ask_llm` 结合上下文回答）。

- 第 12 页：图片与表格检索（多模态）
  - 代码：`image_table.py`
  - 功能：`extract_images_from_pdf` / `summarize_image` / `context_augmentation`；`extract_tables_from_pdf` / `table_context_augmentation`。

- 第 13 页：作业要求（完整 RAG）
  - 组合上述模块：索引 -> 处理 -> 检索 -> 融合/重排 -> 生成回答；可扩展 RAG Fusion/指代/分解。


