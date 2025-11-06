#!/usr/bin/env python3
"""
RAG 系统主入口：
- 统一的启动界面
- 提供菜单选项：导入文件、查看文档、问答等
"""

import os
import sys
import argparse
from config import get_es
from es_functions import create_elastic_index
from batch_ingest import batch_ingest_pdfs
from interactive_qa import interactive_qa
from retrieve_documents import elastic_search

# 自动加载配置文件
def load_config():
    """自动加载配置文件（.env_qwen 或 .env）"""
    from dotenv import load_dotenv
    
    # 优先加载 .env_qwen（通义千问专用配置）
    if os.path.exists('.env_qwen'):
        # 读取 .env_qwen 文件并设置环境变量
        with open('.env_qwen', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # 解析 export VAR=value 或 VAR=value 格式
                if 'OPENAI_API_BASE_URL=' in line:
                    if 'export' in line:
                        value = line.split('OPENAI_API_BASE_URL=')[1].strip().strip('"\'')
                    else:
                        value = line.split('OPENAI_API_BASE_URL=')[1].strip().strip('"\'')
                    os.environ['OPENAI_API_BASE_URL'] = value
                elif 'OPENAI_API_KEY=' in line:
                    if 'export' in line:
                        value = line.split('OPENAI_API_KEY=')[1].strip().strip('"\'')
                    else:
                        value = line.split('OPENAI_API_KEY=')[1].strip().strip('"\'')
                    os.environ['OPENAI_API_KEY'] = value
        print("[配置] 已自动加载 .env_qwen 配置文件")
    else:
        # 如果没有 .env_qwen，尝试加载标准 .env 文件
        load_dotenv()
        if os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_KEY"):
            print("[配置] 已加载 .env 配置文件")

# 启动时自动加载配置
load_config()

def show_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("RAG 系统 - 主菜单")
    print("=" * 60)
    print("1. 导入 PDF 文件（批量入库）")
    print("2. 查看已导入的文档")
    print("3. 开始问答")
    print("4. 快速检索测试")
    print("5. 退出")
    print("=" * 60)

def import_pdfs(index_name: str, pdf_path: str = None):
    """导入 PDF 文件"""
    print("\n" + "=" * 60)
    print("导入 PDF 文件")
    print("=" * 60)
    
    if not pdf_path:
        pdf_path = input("请输入 PDF 文件路径或文件夹路径（默认: test_pdf/）: ").strip()
        if not pdf_path:
            pdf_path = "test_pdf/"
    
    if not os.path.exists(pdf_path):
        print(f"[错误] {pdf_path} 不存在")
        return
    
    # 询问是否处理图片和表格（统一询问）
    process_images = True
    process_tables = True
    use_local = False
    
    if os.path.isdir(pdf_path):
        choice = input("\n是否处理图片和表格？(y/n，默认: y): ").strip().lower()
        if choice == 'n':
            process_images = False
            process_tables = False
    
    local_choice = input("使用本地 embedding 模型？(y/n，默认: n): ").strip().lower()
    use_local = (local_choice == 'y')
    
    print("\n开始批量入库...")
    batch_ingest_pdfs(
        index_name=index_name,
        pdf_path=pdf_path,
        process_images=process_images,
        process_tables=process_tables,
        use_local=use_local
    )
    
    input("\n按 Enter 键返回主菜单...")

def show_documents(index_name: str):
    """查看已导入的文档"""
    print("\n" + "=" * 60)
    print("已导入的文档")
    print("=" * 60)
    
    es = get_es()
    
    if not es.indices.exists(index=index_name):
        print(f"[错误] 索引 {index_name} 不存在")
        input("\n按 Enter 键返回主菜单...")
        return
    
    # 统计信息
    count = es.count(index=index_name)['count']
    print(f"\n[索引] {index_name}")
    print(f"[总数] {count} 条文档")
    
    # 按类型统计
    try:
        text_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "text"}}})['count']
        image_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "image"}}})['count']
        table_count = es.count(index=index_name, body={"query": {"term": {"doc_type": "table"}}})['count']
        
        print(f"\n[文档类型统计]")
        print(f"  - 文本块: {text_count}")
        print(f"  - 图片: {image_count}")
        print(f"  - 表格: {table_count}")
    except:
        pass
    
    # 显示所有 PDF 文件
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
            print(f"\n[参考文档] 共 {len(unique_files)} 个 PDF 文件：")
            for i, file_name in enumerate(unique_files, 1):
                file_count = es.count(
                    index=index_name,
                    body={"query": {"term": {"file_name": file_name}}}
                )['count']
                print(f"  {i}. {file_name} ({file_count} 条文档)")
        else:
            print("\n[提示] 索引中暂无文档，请先导入 PDF")
    except Exception as e:
        print(f"[提示] 无法获取文档列表: {e}")
    
    input("\n按 Enter 键返回主菜单...")

def quick_search(index_name: str):
    """快速检索测试"""
    print("\n" + "=" * 60)
    print("快速检索测试")
    print("=" * 60)
    
    query = input("请输入查询问题: ").strip()
    if not query:
        print("[错误] 查询不能为空")
        input("\n按 Enter 键返回主菜单...")
        return
    
    top_k = input("返回前 K 个结果（默认: 5）: ").strip()
    top_k = int(top_k) if top_k.isdigit() else 5
    
    print(f"\n[检索] 查询: {query}")
    print("正在检索...\n")
    
    try:
        results = elastic_search(query, index_name)
        
        if not results:
            print("未找到相关文档")
        else:
            print(f"找到 {len(results)} 个结果，显示前 {top_k} 个：\n")
            for i, r in enumerate(results[:top_k], 1):
                preview = (r.get("text") or "")[:200].replace("\n", " ")
                file_name = r.get("file_name", "未知文件")
                page = r.get("page")
                ref = f"[{file_name}"
                if page:
                    ref += f" - p.{page}"
                ref += "]"
                print(f"#{i} {ref}")
                print(f"   {preview}...")
                print()
    except Exception as e:
        print(f"[错误] 检索失败: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按 Enter 键返回主菜单...")

def main_loop(index_name: str, use_rerank: bool = False, use_local: bool = False):
    """主循环"""
    while True:
        show_menu()
        choice = input("请选择 (1-5): ").strip()
        
        if choice == '1':
            import_pdfs(index_name)
        elif choice == '2':
            show_documents(index_name)
        elif choice == '3':
            print("\n" + "=" * 60)
            print("启动交互式问答系统")
            print("=" * 60)
            try:
                interactive_qa(index_name, use_rerank=use_rerank, use_local=use_local)
            except KeyboardInterrupt:
                print("\n\n返回主菜单...")
            except Exception as e:
                print(f"\n[错误] {e}")
                import traceback
                traceback.print_exc()
                input("\n按 Enter 键返回主菜单...")
        elif choice == '4':
            quick_search(index_name)
        elif choice == '5':
            print("\n再见！")
            sys.exit(0)
        else:
            print("\n[错误] 无效选择，请输入 1-5")

def main():
    parser = argparse.ArgumentParser(description="RAG 系统主入口")
    parser.add_argument("--index", required=True, help="Elasticsearch 索引名")
    parser.add_argument("--rerank", action="store_true", help="问答时使用 reranker")
    parser.add_argument("--use-local", action="store_true", help="使用本地 embedding 模型")
    parser.add_argument("--auto-import", help="启动时自动导入 PDF（指定路径）")
    args = parser.parse_args()
    
    es = get_es()
    
    # 确保索引存在
    if not es.indices.exists(index=args.index):
        print(f"[创建索引] {args.index}")
        create_elastic_index(args.index)
    
    # 如果指定了自动导入
    if args.auto_import:
        print(f"[自动导入] {args.auto_import}")
        batch_ingest_pdfs(
            index_name=args.index,
            pdf_path=args.auto_import,
            process_images=True,
            process_tables=True,
            use_local=args.use_local
        )
        print("\n自动导入完成！")
    
    # 显示欢迎信息
    print("\n欢迎使用 RAG 系统！")
    print(f"索引: {args.index}")
    
    # 显示 LLM 配置状态
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url and api_key:
        print(f"[LLM] 已配置: {base_url.split('/')[-2] if '/' in base_url else '已配置'}")
    else:
        print("[LLM] 未配置（将使用简单拼接方式）")
    
    # 显示当前状态
    count = es.count(index=args.index)['count']
    if count > 0:
        print(f"[索引] 当前索引包含 {count} 条文档")
    else:
        print("[提示] 索引为空，请先导入 PDF 文件")
    
    # 启动主循环
    try:
        main_loop(args.index, use_rerank=args.rerank, use_local=args.use_local)
    except KeyboardInterrupt:
        print("\n\n再见！")
        sys.exit(0)

if __name__ == "__main__":
    main()

