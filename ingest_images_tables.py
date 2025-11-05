"""
图片与表格入库模块：
- 从 PDF 提取图片/表格并生成描述
- 将描述向量化并写入 ES，建立映射关系（image_id/table_id）
- 文字被命中时可召回原图/原表
"""

from config import get_es
from embedding import local_embedding
from image_table import extract_images_from_pdf, extract_tables_from_pdf
import os


def ingest_images(index_name: str, pdf_path: str):
    """提取图片描述并入库（向量化+写入ES+建立映射）。"""
    es = get_es()
    file_name = os.path.basename(pdf_path)
    
    # 提取图片并生成描述
    image_results = extract_images_from_pdf(pdf_path)
    
    if not image_results:
        print(f"[Images] No images found in {pdf_path}")
        return
    
    # 批量向量化
    descriptions = [img.get("context_augmented_summary") or img.get("summary") or "" 
                    for img in image_results]
    descriptions = [d for d in descriptions if d and d != "0"]  # 过滤无效描述
    
    if not descriptions:
        print(f"[Images] No valid descriptions generated")
        return
    
    vectors = local_embedding(descriptions, use_local=False)
    
    # 写入 ES（建立映射）
    for img, desc, vec in zip(image_results, descriptions, vectors):
        page_num = img.get("page_num", 0)
        image_id = f"{file_name}_img_p{page_num + 1}_idx{img.get('image_index', 1)}"
        
        body = {
            "text": desc,
            "vector": vec,
            "file_name": file_name,
            "page": page_num + 1,
            "doc_type": "image",
            "image_id": image_id
        }
        es.index(index=index_name, body=body)
    
    print(f"[Images] Ingested {len(descriptions)} image descriptions into {index_name}")


def ingest_tables(index_name: str, pdf_path: str):
    """提取表格描述并入库（向量化+写入ES+建立映射）。"""
    es = get_es()
    file_name = os.path.basename(pdf_path)
    
    # 提取表格并生成描述
    table_results = extract_tables_from_pdf(pdf_path)
    
    if not table_results:
        print(f"[Tables] No tables found in {pdf_path}")
        return
    
    # 批量向量化（使用增强后的描述）
    descriptions = [tbl.get("context_augmented_table") or "" 
                    for tbl in table_results]
    descriptions = [d for d in descriptions if d.strip()]
    
    if not descriptions:
        print(f"[Tables] No valid descriptions generated")
        return
    
    vectors = local_embedding(descriptions, use_local=False)
    
    # 写入 ES（建立映射）
    for tbl, desc, vec in zip(table_results, descriptions, vectors):
        page_num = tbl.get("page_num", 0)
        table_id = f"{file_name}_table_p{page_num + 1}_idx{tbl.get('table_index', 1)}"
        
        # 可选：将 Markdown 也存储（用于召回时展示）
        table_md = tbl.get("table_markdown", "")
        
        body = {
            "text": desc,  # 存储增强后的描述（用于检索）
            "vector": vec,
            "file_name": file_name,
            "page": page_num + 1,
            "doc_type": "table",
            "table_id": table_id
            # 可选：如果需要召回时展示表格，可加 "table_markdown": table_md
        }
        es.index(index=index_name, body=body)
    
    print(f"[Tables] Ingested {len(descriptions)} table descriptions into {index_name}")


if __name__ == "__main__":
    # 示例：入库图片和表格
    index = "test_index_1"
    pdf = "test_pdf/image_extraction_example.pdf"
    
    # ingest_images(index, pdf)
    # ingest_tables(index, "test_pdf/table_extraction_example.pdf")
    pass

