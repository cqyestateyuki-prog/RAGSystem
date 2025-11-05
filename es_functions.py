"""
Elasticsearch 索引管理：
- 创建/删除用于 RAG 的向量索引
- 索引包含原文 `text` 与向量 `vector` 字段（1024 维，cosine 相似度）

关于元数据字段（只在需要时开启）：
- 作用：用于检索过滤与来源引用（不参与向量相似度计算）。
- 常见字段及推荐类型：
  - file_name: 关键词过滤/引用来源 → `keyword`
  - page: PDF 页码 → `integer`
  - chapter/section/title: 章节/标题 → `keyword`（若需分词可用 `text`）
  - doc_type/language/tags: 文档类型/语言/标签 → `keyword`
  - file_id: 内部唯一标识 → `long`

开启步骤（不改本文件逻辑，仅作说明）：
1) 取消下方注释并选择合适类型（`keyword` 用于精确匹配，`text` 用于分词搜索）。
2) 重建索引（删旧建新），否则映射无法在线修改。
3) 入库时在文档 body 中写入对应字段值（例如 `file_name`/`page`）。
4) 检索时在 `query` 或 `knn` 的 `filter` 段加入 `term`/`range` 进行过滤。
"""

from config import get_es

def create_elastic_index(index_name):
    """创建用于向量检索的 ES 索引，包含文本与 dense_vector 字段。"""
    es=get_es()
    mappings = {
                "properties": {
                    "text": {
                        "type": "text"
                    }, 
                    "vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                        },
                    # metadata filtering（已启用，用于通用引用与过滤）
                    "file_name": {"type": "keyword"},   # 引用/按文件过滤（精确匹配）
                    "page": {"type": "integer"},        # PDF 页码（便于按页过滤）
                    "doc_type": {"type": "keyword"},    # 文档类型：text/image/table
                    "image_id": {"type": "keyword"},    # 图片ID（用于映射到原图）
                    "table_id": {"type": "keyword"}     # 表格ID（用于映射到原表）
                    # "chapter": {"type": "keyword"},     # 章节（精确匹配）；如需分词可改为 text
                    # "language": {"type": "keyword"},     # 语言（zh/en）
                    # "file_id": {"type": "long"},        # 内部唯一 ID

                    }
                }
    #创建elastic
    try:
        es.indices.create(index=index_name, mappings=mappings)
        print('[Create Vector DB]' + index_name + ' created')
    except Exception as e:
        print(f'Create Vector DB Exception: {e}')

def delete_elastic_index(index_name):
    """删除指定索引（危险操作，谨慎使用）。"""
    es=get_es()
    es.indices.delete(index=index_name)
    print('[Delete Vector DB]' + index_name + ' deleted')

if __name__ == '__main__':
    create_elastic_index('test_index_1')
    # delete_elastic_index('test')