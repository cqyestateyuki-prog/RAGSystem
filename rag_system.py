#!/usr/bin/env python3
"""
交互式问答系统（统一入口）：
- PDF 处理完成后，可以持续提问
- 使用 LLM 生成答案，并显示引用来源
"""

import sys
from retrieve_documents import elastic_search, rerank, rag_fusion
from openai import OpenAI  # 注意：虽然叫 OpenAI，但可以用于任何 OpenAI 兼容接口（包括 Qwen、DeepSeek 等）
from config import get_es
from es_functions import create_elastic_index
import os

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
                for var_name in ['OPENAI_API_BASE_URL', 'OPENAI_API_KEY']:
                    if f'{var_name}=' in line:
                        # 提取值（支持 export VAR=value 或 VAR=value 格式）
                        value = line.split(f'{var_name}=')[1].strip().strip('"\'')
                        os.environ[var_name] = value
                        break
        print("[配置] 已自动加载 .env_qwen 配置文件")
    else:
        # 如果没有 .env_qwen，尝试加载标准 .env 文件
        load_dotenv()
        if os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_API_KEY"):
            print("[配置] 已加载 .env 配置文件")

# 启动时自动加载配置
load_config()

# 初始化 LLM 客户端（用于生成答案）
# 支持多种模型：通义千问（Qwen）、DeepSeek、Ollama 等
# 注意：这些服务都提供 OpenAI 兼容接口，所以可以使用同一个 SDK
try:
    # 优先使用环境变量配置（已自动加载）
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    
    # 如果设置了 base_url，使用自定义服务
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
    elif api_key:
        # 如果只有 api_key，使用默认 OpenAI 配置
        client = OpenAI(api_key=api_key)
    else:
        # 尝试使用 DeepSeek（如果配置了）
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=deepseek_key
            )
        else:
            client = None  # 没有配置任何模型
            print("[提示] 未配置 LLM，将使用简单拼接方式生成答案")
            print("[提示] 配置方法（推荐通义千问）：")
            print("  通义千问（阿里云）:")
            print("    export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1")
            print("    export OPENAI_API_KEY=你的_qwen_api_key")
            print("  或运行: ./config_qwen.sh")
            print("  说明：虽然使用 openai 库，但实际连接的是阿里云通义千问服务")
            print("  详细说明见 LLM_MODELS.md")
except Exception as e:
    print(f"[警告] LLM 客户端初始化失败: {e}")
    client = None

def _format_source(file_name: str, page: int = None) -> str:
    """格式化来源字符串（文件名+页码）"""
    if page:
        return f"{file_name}，第{page}页"
    return file_name

def _build_simple_answer(retrieved_docs: list, top_k: int) -> tuple[str, list]:
    """构建简单拼接答案（当 LLM 不可用时使用）"""
    answer_parts = []
    doc_sources = []
    
    for doc in retrieved_docs[:top_k]:
        file_name = doc.get('file_name', '未知文件')
        page = doc.get('page')
        source = _format_source(file_name, page)
        
        answer_parts.append(f"[{source}]\n{doc.get('text', '')[:300]}")
        doc_sources.append({
            "source": source,
            "file_name": file_name,
            "page": page,
            "text": doc.get("text", "")[:200]
        })
    
    return "\n\n".join(answer_parts), doc_sources

def generate_answer_with_citations(query: str, retrieved_docs: list, top_k: int = 5):
    """使用 LLM 生成答案，并提取引用来源"""
    
    # 构建上下文和来源映射
    context_parts = []
    doc_sources = []
    source_map = {}  # 映射：序号 -> 真实来源
    
    for i, doc in enumerate(retrieved_docs[:top_k], 1):
        text = doc.get("text", "")
        file_name = doc.get("file_name", "未知文件")
        page = doc.get("page")
        
        # 构建真实来源标识（文件名+页码）
        real_source = _format_source(file_name, page)
        
        # 在上下文中使用序号（便于LLM引用），但记录真实映射
        doc_id = f"[来源{i}]"
        source_map[doc_id] = real_source
        source_map[f"来源{i}"] = real_source
        source_map[f"文档{i}"] = real_source  # 兼容旧格式
        
        context_parts.append(f"{doc_id} {real_source}\n{text}")
        
        doc_sources.append({
            "source": real_source,  # 使用真实来源
            "file_name": file_name,
            "page": page,
            "text": text[:200]  # 预览
        })
    
    context = "\n\n".join(context_parts)
    
    # 构建 prompt，明确要求使用文件名+页码格式
    prompt = f"""基于以下检索到的文档内容，回答用户的问题。请确保答案准确、完整，并在适当位置引用来源。

检索到的文档内容：
{context}

用户问题：{query}

请生成答案，并在答案中标注引用来源。引用格式必须使用：文件名+页码（例如：来源: 刑事诉讼法.pdf，第3页）。
如果多个来源，格式为：来源: 文件名1，第X页；文件名2，第Y页。
不要编造答案或使用文档中没有的信息。

重要提示：
- 如果检索到的文档内容与用户问题不相关，请明确说明"根据检索到的文档，无法回答此问题"或"检索到的文档内容与您的问题不相关"
- 不要编造答案或使用文档中没有的信息
- 不要对不相关的问题给出通用回答（如"你好"、"谢谢"等）"""
    
    # 调用 LLM 生成答案
    if client is None:
        # 如果客户端未初始化，使用简单拼接（使用真实来源）
        print("[提示] 未配置 LLM，使用简单拼接方式生成答案...")
        answer, doc_sources = _build_simple_answer(retrieved_docs, top_k)
    else:
        try:
            # 尝试多个模型（根据 base_url 自动选择）
            base_url = os.getenv("OPENAI_API_BASE_URL", "")
            if "volces.com" in base_url or os.getenv("DEEPSEEK_API_KEY"):
                # DeepSeek
                models_to_try = ["deepseek-v3-250324", "deepseek-chat"]
            elif "dashscope" in base_url:
                # 通义千问
                models_to_try = ["qwen-plus", "qwen-max", "qwen-turbo"]
            elif "localhost:11434" in base_url or "ollama" in base_url.lower():
                # Ollama
                models_to_try = ["qwen2.5:7b", "llama3.2:3b", "mistral"]
            else:
                # 默认 OpenAI 或其他兼容服务
                models_to_try = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
            response = None
            
            for model in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "你是一个智能助手，基于检索到的文档内容回答问题。引用来源时必须使用完整的文件名和页码格式（例如：刑事诉讼法.pdf，第3页），不要使用'文档1'、'文档2'这样的序号。如果检索到的文档内容与用户问题不相关，请明确说明'根据检索到的文档，无法回答此问题'或'检索到的文档内容与您的问题不相关'，不要对不相关的问题给出通用回答。不要编造答案或使用文档中没有的信息。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    break
                except Exception as e:
                    if model == models_to_try[-1]:
                        raise e
                    continue
            
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"[警告] LLM 调用失败: {e}")
            print("使用简单拼接方式生成答案...")
            # 回退到简单拼接（使用真实来源）
            answer, doc_sources = _build_simple_answer(retrieved_docs, top_k)
    
    return answer, doc_sources

def interactive_qa(index_name: str, use_rerank: bool = False, use_local: bool = False):
    """交互式问答循环"""
    while True:
        # 提示输入
        print("=" * 60)
        print("输入您想了解的问题 (输入 'exit' 退出AI问答系统): ", end="", flush=True)
        query = input().strip()
        
        if query.lower() in ['exit', 'quit', '退出', 'q']:
            print("\n再见！")
            break
        
        if not query:
            continue
        
        print(f"你的问题: {query}")
        print("正在检索和生成答案...")
        print()
        
        # 检索文档
        # 提示：确保使用的 embedding 模型与索引创建时一致
        if use_local:
            print("[检索] 使用本地 embedding 模型")
        else:
            print("[检索] 使用远程 embedding 服务")
        results = elastic_search(query, index_name, use_local=use_local)
        
        # 可选：使用 rerank
        if use_rerank:
            results = rerank(query, results[:20])
        
        if not results:
            print("未找到相关文档，请尝试其他问题。")
            print()
            continue
        
        # 生成答案
        answer, doc_sources = generate_answer_with_citations(query, results, top_k=5)
        
        # 显示答案
        print("=" * 60)
        print("AI 回答:")
        print("=" * 60)
        print(answer)
        
        # 显示检索到的文档（基于 file_name + page 去重，每个来源只显示一次）
        print()
        print("=" * 60)
        
        # 去重：基于 file_name + page 的组合，保留第一个出现的（最相关的）
        seen_sources = set()
        unique_doc_sources = []
        for doc in doc_sources:
            source_key = doc['source']  # file_name + page 的组合
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_doc_sources.append(doc)
            if len(unique_doc_sources) >= 5:
                break
        
        print(f"【参考文档】共 {len(unique_doc_sources)} 个:")
        for i, doc in enumerate(unique_doc_sources, 1):
            print(f"  {i}. {doc['source']}")
            # 显示文本预览（如果存在）
            text = doc.get('text', '').strip()
            if text:
                # 去掉文本开头的页码（如果存在）
                page = doc.get('page')
                if page and text.startswith(str(page)):
                    text = text[len(str(page)):].lstrip('\n\r\t ')
                if text:
                    print(f"     {text}...")
        print()

def _list_indices(es):
    try:
        indices = es.indices.get_alias(index='*,-.*')
    except Exception:
        indices = es.indices.get_alias(index='*')
    if not indices:
        print("No indices found")
        return
    user_index_names = [n for n in indices.keys() if not n.startswith('.')]
    if not user_index_names:
        print("No user indices found (only system indices present)")
        return
    print("Available indices:")
    for name in sorted(user_index_names):
        try:
            count = es.count(index=name)['count']
            print(f"- {name}: {count}")
        except Exception:
            print(f"- {name}")


def _ensure_index(es, index_name: str):
    if not es.indices.exists(index=index_name):
        create_elastic_index(index_name)
    else:
        print(f"Index \"{index_name}\" already exists")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="交互式 RAG 系统（索引->处理->检索->融合/重排->回答）")
    parser.add_argument("--index", help="Elasticsearch 索引名（提供则直接进入问答）")
    parser.add_argument("--rerank", action="store_true", help="使用 reranker 重排序")
    parser.add_argument("--use-local", action="store_true", help="使用本地 embedding 模型")
    args = parser.parse_args()

    es = get_es()

    if args.index:
        # 检查索引是否存在
        if not es.indices.exists(index=args.index):
            print(f"[错误] 索引 \"{args.index}\" 不存在")
            print("[提示] 请先创建索引并添加PDF文档")
            print("  使用菜单系统：python rag_system.py")
            print("  或使用命令行：python batch_ingest.py --index {} --pdf <PDF路径>".format(args.index))
            sys.exit(1)
        # 检查索引是否有数据
        try:
            count = es.count(index=args.index)['count']
            if count == 0:
                print(f"[警告] 索引 \"{args.index}\" 存在但没有数据")
                print("[提示] 请先添加PDF文档")
                print("  使用菜单系统：python rag_system.py，选择选项3")
                print("  或使用命令行：python batch_ingest.py --index {} --pdf <PDF路径>".format(args.index))
                sys.exit(1)
            print(f"\n[索引] {args.index} 包含 {count} 条文档")
        except Exception:
            pass
        try:
            interactive_qa(args.index, use_rerank=args.rerank, use_local=args.use_local)
        except KeyboardInterrupt:
            print("\n\n再见！")
        return

    from es_functions import delete_elastic_index
    from batch_ingest import batch_ingest_pdfs
    from ingest_images_tables import ingest_images, ingest_tables
    from websearch import bocha_web_search, ask_llm

    while True:
        print("\n=== PDF RAG System ===")
        print("Options:")
        print("1. Create index")
        print("2. Delete index")
        print("3. Add PDF to existing index")
        print("4. Search in existing index")
        print("5. List indices")
        print("6. Start QA in index")
        print("7. Web search QA")
        print("8. RAG Fusion QA (multi-query)")
        print("9. Exit")
        choice = input("\nEnter choice (1-9): ").strip()

        if choice == '1':
            name = input("Enter index name: ").strip()
            if not name:
                continue
            _ensure_index(es, name)
            input("\nPress Enter to go back to Options")
        elif choice == '2':
            name = input("Enter index name to delete: ").strip()
            if not name:
                continue
            try:
                if es.indices.exists(index=name):
                    confirm = input(f"确认删除索引 \"{name}\"? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        delete_elastic_index(name)
                    else:
                        print("取消删除")
                else:
                    print(f"Index \"{name}\" does not exist")
            except Exception as e:
                print(f"[错误] 删除失败: {e}")
            input("\nPress Enter to go back to Options")
        elif choice == '3':
            name = input("Enter index name: ").strip()
            if not name:
                continue
            _ensure_index(es, name)
            pdf_path = input("Enter PDF file path or folder path: ").strip()
            if not pdf_path or not os.path.exists(pdf_path):
                print(f"[错误] 路径不存在: {pdf_path}")
                input("\nPress Enter to go back to Options")
                continue
            try:
                process_images = input("Process images? (y/n, default=y): ").strip().lower() != 'n'
                process_tables = input("Process tables? (y/n, default=y): ").strip().lower() != 'n'
                use_local = input("Use local embedding model? (y/n, default=n): ").strip().lower() == 'y'
                batch_ingest_pdfs(name, pdf_path, process_images, process_tables, use_local)
            except Exception as e:
                print(f"[错误] 入库失败: {e}")
                import traceback
                traceback.print_exc()
            input("\nPress Enter to go back to Options")
        elif choice == '4':
            name = input("Enter index name: ").strip()
            if not name:
                continue
            if not es.indices.exists(index=name):
                print(f"[错误] 索引 \"{name}\" 不存在")
                print("[提示] 请先使用选项1创建索引，或使用选项3添加PDF文档")
                input("\nPress Enter to go back to Options")
                continue
            query = input("Enter search query: ").strip()
            if not query:
                continue
            try:
                # 询问是否使用本地模型（应与索引创建时一致）
                use_local_search = input("Use local embedding model? (y/n, default=n): ").strip().lower() == 'y'
                if use_local_search:
                    print("[检索] 使用本地 embedding 模型")
                else:
                    print("[检索] 使用远程 embedding 服务")
                results = elastic_search(query, name, use_local=use_local_search)
                print(f"\n找到 {len(results)} 条结果:")
                for i, r in enumerate(results[:5], 1):
                    file_name = r.get('file_name', '未知文件')
                    page = r.get('page')
                    ref = f"[{file_name}"
                    if page:
                        ref += f" - p.{page}"
                    ref += "]"
                    print(f"#{i} {ref}")
                    print(f"   {r.get('text', '')[:200]}...")
                    print()
            except Exception as e:
                print(f"[错误] 检索失败: {e}")
                import traceback
                traceback.print_exc()
            input("\nPress Enter to go back to Options")
        elif choice == '5':
            _list_indices(es)
            input("\nPress Enter to go back to Options")
        elif choice == '6':
            name = input("Enter index name to start QA: ").strip()
            if not name:
                continue
            # 检查索引是否存在，不存在则提示用户
            if not es.indices.exists(index=name):
                print(f"[错误] 索引 \"{name}\" 不存在")
                print("[提示] 请先使用选项1创建索引，或使用选项3添加PDF文档")
                input("\nPress Enter to go back to Options")
                continue
            # 检查索引是否有数据
            try:
                count = es.count(index=name)['count']
                if count == 0:
                    print(f"[警告] 索引 \"{name}\" 存在但没有数据")
                    print("[提示] 请先使用选项3添加PDF文档")
                    input("\nPress Enter to go back to Options")
                    continue
            except Exception:
                pass
            # 询问是否使用本地模型（应与索引创建时一致）
            use_local_qa = input("Use local embedding model? (y/n, default=n): ").strip().lower() == 'y'
            try:
                interactive_qa(name, use_rerank=args.rerank, use_local=use_local_qa)
            except KeyboardInterrupt:
                print("\n\n再见！")
        elif choice == '7':
            print("\n=== Web Search QA ===")
            q = input("Enter your question: ").strip()
            if not q:
                continue
            try:
                print("\n[WebSearch] 正在检索网页上下文...")
                webctx = bocha_web_search(q)
                print("[WebSearch] 上下文获取完成，正在调用 LLM 生成答案...\n")
                ans = ask_llm(q, webctx)
                print("=" * 60)
                print("Web 搜索回答:")
                print("=" * 60)
                print(ans)
                print("=" * 60)
            except Exception as e:
                print(f"[错误] Web 搜索问答失败: {e}")
                import traceback
                traceback.print_exc()
            input("\nPress Enter to go back to Options")
        elif choice == '8':
            name = input("Enter index name: ").strip()
            if not name:
                continue
            if not es.indices.exists(index=name):
                print(f"Index \"{name}\" does not exist")
                input("\nPress Enter to go back to Options")
                continue
            q = input("Enter your question: ").strip()
            if not q:
                continue
            try:
                print("\n[Fusion] 生成改写查询...")
                variants = rag_fusion(q) or []
                all_queries = [q] + [v for v in variants if isinstance(v, str)]
                
                # 如果 RAG Fusion 失败，只有原始查询，提示用户
                if len(all_queries) == 1:
                    print(f"[警告] RAG Fusion 查询改写失败，仅使用原始查询。结果将与选项6相同。")
                    print(f"[提示] 请检查 OpenAI API 配置，或使用选项6进行单查询检索。")
                    print(f"[Fusion] 使用 1 条查询：{all_queries}")
                else:
                    print(f"[Fusion] 使用 {len(all_queries)} 条查询：{all_queries}")
                
                k = 60
                fused = {}
                doc_store = {}
                # 询问是否使用本地模型（应与索引创建时一致）
                use_local_fusion = input("Use local embedding model? (y/n, default=n): ").strip().lower() == 'y'
                if use_local_fusion:
                    print("[Fusion] 使用本地 embedding 模型")
                else:
                    print("[Fusion] 使用远程 embedding 服务")
                for sub_q in all_queries:
                    res = elastic_search(sub_q, name, use_local=use_local_fusion)
                    for doc in res[:20]:
                        doc_id = doc.get('id') or f"{doc.get('file_name')}_{doc.get('page')}_{hash(doc.get('text',''))}"
                        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + doc.get('rank', 9999))
                        if doc_id not in doc_store:
                            doc_store[doc_id] = doc
                ranked_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)
                fused_docs = [doc_store[i] for i,_ in ranked_ids]
                answer, doc_sources = generate_answer_with_citations(q, fused_docs, top_k=5)
                print("=" * 60)
                print("AI 回答 (RAG Fusion):")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                print(f"\n【参考文档】共 {min(5, len(doc_sources))} 个:")
                for i, doc in enumerate(doc_sources[:5], 1):
                    print(f"  {i}. {doc['source']}")
                print("=" * 60)
                input("\nPress Enter to go back to Options")
            except Exception as e:
                print(f"[错误] RAG Fusion QA 失败: {e}")
                import traceback
                traceback.print_exc()
        elif choice == '9':
            print("Bye")
            break
        else:
            continue

if __name__ == "__main__":
    main()
