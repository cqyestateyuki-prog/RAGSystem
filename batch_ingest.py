#!/usr/bin/env python3
"""
批量入库脚本：
- 扫描文件夹中的所有 PDF 文件
- 自动处理文本、图片、表格
- 支持指定文件或整个文件夹
"""

import os
import argparse
from pathlib import Path
from run import ingest_pdf
from ingest_images_tables import ingest_images, ingest_tables
from es_functions import create_elastic_index
from config import get_es

def batch_ingest_pdfs(index_name: str, pdf_path: str, process_images: bool = True, 
                     process_tables: bool = True, use_local: bool = False):
    """
    批量入库 PDF 文件
    
    参数：
    - index_name: Elasticsearch 索引名
    - pdf_path: PDF 文件路径或文件夹路径
    - process_images: 是否处理图片
    - process_tables: 是否处理表格
    - use_local: 是否使用本地 embedding 模型
    """
    es = get_es()
    
    # 确保索引存在
    if not es.indices.exists(index=index_name):
        print(f"[创建索引] {index_name}")
        create_elastic_index(index_name)
    else:
        print(f"[索引] {index_name} 已存在")
    
    # 收集所有 PDF 文件
    pdf_files = []
    
    if os.path.isfile(pdf_path):
        # 单个文件
        if pdf_path.lower().endswith('.pdf'):
            pdf_files.append(pdf_path)
        else:
            print(f"[错误] {pdf_path} 不是 PDF 文件")
            return
    elif os.path.isdir(pdf_path):
        # 文件夹，扫描所有 PDF（支持递归）
        for file in Path(pdf_path).rglob("*.pdf"):
            pdf_files.append(str(file))
        print(f"[发现] 在 {pdf_path} 中找到 {len(pdf_files)} 个 PDF 文件")
        if pdf_files:
            print("\n找到的 PDF 文件：")
            for i, f in enumerate(pdf_files, 1):
                print(f"  {i}. {os.path.basename(f)}")
            print()
    else:
        print(f"[错误] {pdf_path} 不存在")
        return
    
    if not pdf_files:
        print("[错误] 未找到 PDF 文件")
        return
    
    # 处理每个 PDF
    total_files = len(pdf_files)
    for idx, pdf_file in enumerate(pdf_files, 1):
        file_name = os.path.basename(pdf_file)
        print("\n" + "=" * 60)
        print(f"[{idx}/{total_files}] 处理: {file_name}")
        print("=" * 60)
        
        try:
            # 1. 文本入库
            print(f"\n[1/3] 文本入库...")
            ingest_pdf(index_name, pdf_file, use_local=use_local)
            
            # 2. 图片入库（可选）
            if process_images:
                print(f"\n[2/3] 图片入库...")
                try:
                    ingest_images(index_name, pdf_file)
                except Exception as e:
                    print(f"[警告] 图片处理失败: {e}")
            
            # 3. 表格入库（可选）
            if process_tables:
                print(f"\n[3/3] 表格入库...")
                try:
                    ingest_tables(index_name, pdf_file)
                except Exception as e:
                    print(f"[警告] 表格处理失败: {e}")
            
            print(f"\n✅ {file_name} 处理完成")
            
        except Exception as e:
            print(f"\n❌ {file_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 统计信息
    print("\n" + "=" * 60)
    print("批量入库完成！")
    print("=" * 60)
    
    # 检查索引统计
    count = es.count(index=index_name)['count']
    print(f"\n[统计] 索引 {index_name} 中共有 {count} 条文档")
    
    # 按类型统计
    text_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "text"}}})['count']
    image_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "image"}}})['count']
    table_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "table"}}})['count']
    
    print(f"  - 文本块: {text_count}")
    print(f"  - 图片: {image_count}")
    print(f"  - 表格: {table_count}")
    
    # 显示所有参考的 PDF 文件
    try:
        agg_result = es.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    "unique_files": {
                        "terms": {
                            "field": "file_name",
                            "size": 100
                        }
                    }
                }
            }
        )
        
        unique_files = [bucket["key"] for bucket in agg_result["aggregations"]["unique_files"]["buckets"]]
        if unique_files:
            print(f"\n[参考文档] 索引中包含 {len(unique_files)} 个 PDF 文件：")
            for file_name in unique_files:
                file_count = es.count(
                    index=index_name,
                    body={"query": {"term": {"file_name": file_name}}}
                )['count']
                print(f"  - {file_name} ({file_count} 条文档)")
    except Exception as e:
        pass  # 忽略聚合查询错误


def main():
    parser = argparse.ArgumentParser(description="批量入库 PDF 文件到 Elasticsearch")
    parser.add_argument("--index", required=True, help="Elasticsearch 索引名")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径或包含 PDF 的文件夹路径")
    parser.add_argument("--no-images", action="store_true", help="跳过图片处理")
    parser.add_argument("--no-tables", action="store_true", help="跳过表格处理")
    parser.add_argument("--use-local", action="store_true", help="使用本地 embedding 模型")
    args = parser.parse_args()
    
    batch_ingest_pdfs(
        index_name=args.index,
        pdf_path=args.pdf,
        process_images=not args.no_images,
        process_tables=not args.no_tables,
        use_local=args.use_local
    )


if __name__ == "__main__":
    main()

