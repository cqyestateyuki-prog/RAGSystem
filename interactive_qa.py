#!/usr/bin/env python3
"""
交互式问答系统：
- PDF 处理完成后，可以持续提问
- 使用 LLM 生成答案，并显示引用来源
"""

import sys
from retrieve_documents import elastic_search, rerank
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
不要编造答案或使用文档中没有的信息。"""
    
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
                            {"role": "system", "content": "你是一个智能助手，基于检索到的文档内容回答问题。引用来源时必须使用完整的文件名和页码格式（例如：刑事诉讼法.pdf，第3页），不要使用'文档1'、'文档2'这样的序号。不要编造答案或使用文档中没有的信息。"},
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
        print("输入问题 (输入 'exit' 退出): ", end="", flush=True)
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
        
        # 显示检索到的文档（显示前5个，不去重）
        print(f"\n【参考文档】共 {len(doc_sources)} 个:")
        for i, doc in enumerate(doc_sources[:5], 1):
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="交互式 RAG 问答系统")
    parser.add_argument("--index", required=True, help="Elasticsearch 索引名")
    parser.add_argument("--rerank", action="store_true", help="使用 reranker 重排序")
    parser.add_argument("--use-local", action="store_true", help="使用本地 embedding 模型")
    args = parser.parse_args()
    
    # 确保索引存在
    es = get_es()
    if not es.indices.exists(index=args.index):
        print(f"[创建索引] {args.index}")
        create_elastic_index(args.index)
    
    # 检查索引是否有数据
    count = es.count(index=args.index)['count']
    if count == 0:
        print(f"[警告] 索引 {args.index} 中没有数据，请先入库 PDF 文档。")
        print("使用: python batch_ingest.py --index {} --pdf <PDF文件夹路径>".format(args.index))
        sys.exit(1)
    
    # 显示索引中的文档来源
    print(f"\n[索引] {args.index} 包含 {count} 条文档")
    
    # 查询所有不同的文件来源
    try:
        # 使用聚合查询获取所有不同的 file_name
        agg_result = es.search(
            index=args.index,
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
            print(f"[参考文档] 共 {len(unique_files)} 个 PDF 文件")
            for i, file_name in enumerate(unique_files, 1):
                # 统计每个文件的文档数
                file_count = es.count(
                    index=args.index,
                    body={"query": {"term": {"file_name": file_name}}}
                )['count']
                print(f"  {i}. {file_name} ({file_count} 条文档)")
    except Exception as e:
        pass  # 静默失败，不影响使用
    
    # 启动交互式问答
    try:
        interactive_qa(args.index, use_rerank=args.rerank, use_local=args.use_local)
    except KeyboardInterrupt:
        print("\n\n再见！")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

