"""
检索与增强相关功能模块：
- 提供混合搜索（关键词 + 向量）并用 RRF 融合结果
- 提供重排序（rerank）接口以提升最终排序质量
- 提供 RAG Fusion（查询改写）、指代消解、查询拆分等高级能力

核心入口：elastic_search(text, es_index)
返回：融合排序后的文档片段列表（包含 id、text、rank 等）
"""

import json
from openai import OpenAI
from config import get_es
from embedding import local_embedding
import jieba
import re
import requests
from config import RERANK_URL

# 全局 reranker 变量（用于懒加载）
_reranker_model = None
_reranker_tokenizer = None
def elastic_search(text, es_index, use_local: bool = False):
    """执行混合检索（关键词 + 向量）并用 RRF 融合结果。

    参数：
    - text: 用户查询文本
    - es_index: Elasticsearch 索引名
    - use_local: 是否使用本地模型（True=本地，False=远程服务，默认False）

    返回：
    - 按融合分数排序的文档结果列表，每个元素含 id/text/metadata/rank 等
    """
    es=get_es()
    key_words = get_keyword(text)


    # 关键词检索查询：使用 should + fuzziness 允许一定模糊匹配
    keyword_query = {
        "bool": {
            "should": [
                {"match": {"text": {"query": keyword, "fuzziness": "AUTO"}}} for keyword in key_words
            ],
            "minimum_should_match": 1
        }
    }
    res_keyword = es.search(index=es_index, query=keyword_query)
    keyword_hits = [{'id': hit['_id'], 'text': hit['_source'].get('text'), 
                     'file_name': hit['_source'].get('file_name'),
                     'page': hit['_source'].get('page'),
                     'doc_type': hit['_source'].get('doc_type'),
                     'image_id': hit['_source'].get('image_id'),
                     'table_id': hit['_source'].get('table_id'),
                     'rank': idx + 1} for idx, hit in enumerate(res_keyword['hits']['hits'])]
    # print(keyword_hits)
    # keyword_hits = [] #test vector search

    # 向量检索查询：通过 script_score 使用余弦相似度（与索引中 dense_vector 一致）
    embedding = local_embedding([text], use_local=use_local)
    vector_query = {
        "bool": {
            "must": [{"match_all": {}}],
            "should": [
                {"script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                        "params": {"queryVector": embedding[0]}
                    }
                }}
            ]
        }
    }
    res_vector = es.search(index=es_index, query=vector_query)
    
    # print(res_vector)
    
    vector_hits = [{'id': hit['_id'], 'text': hit['_source'].get('text'), 
                    'file_name': hit['_source'].get('file_name'),
                    'page': hit['_source'].get('page'),
                    'doc_type': hit['_source'].get('doc_type'),
                    'image_id': hit['_source'].get('image_id'),
                    'table_id': hit['_source'].get('table_id'),
                    'rank': idx + 1} for idx, hit in enumerate(res_vector['hits']['hits'])]
    
    # print(vector_hits)
    # 使用 RRF 将关键词与向量检索结果进行融合，兼顾精确匹配与语义匹配
    combined_results = hybrid_search_rrf(keyword_hits, vector_hits)
    # print(combined_results)
    return combined_results

def get_keyword(query):
    """对中文查询进行分词，过滤停用词，产出用于 BM25 的关键词列表。"""
    # 确保输入是字符串类型
    if not isinstance(query, str):
        print(f'[Get Keyword] Received non-string query: {query} (type: {type(query)}), converting to string')
        query = str(query) if query is not None else ''
    
    # 确保查询不为空
    if not query.strip():
        print('[Get Keyword] Empty query string, returning empty list')
        return []
    
    try:
        # 使用搜索引擎模式进行分词
        seg_list = jieba.cut_for_search(query)
        # Filter out stop words
        filtered_keywords = [word for word in seg_list if word not in stop_words]
        # logging.info('[Jieba Keywords Extraction] ' + ','.join(filtered_keywords))
        return filtered_keywords
    except Exception as e:
        print(f'[Get Keyword] Error processing query "{query}": {e}')
        return []
    
stop_words = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "与", "如何",
    "为", "得", "里", "后", "自己", "之", "过", "给", "然后", "那", "下", "能", "而", "来", "个", "这", "之间", "应该", "可以", "到", "由", "及", "对", "中", "会",
    "但", "年", "还", "并", "如果", "我们", "为了", "而且", "或者", "因为", "所以", "对于", "而言", "与否", "只是", "已经", "可能", "同时", "比如", "这样", "当然",
    "并且", "大家", "之后", "那么", "越", "虽然", "比", "还是", "只有", "现在", "应该", "由于", "尽管", "除了", "以外", "然而", "哪些", "这些", "所有", "并非",
    "例如", "尤其", "哪里", "那里", "何时", "多少", "以至", "以至于", "几乎", "已经", "仍然", "甚至", "更加", "无论", "不过", "不是", "从来", "何处", "到底", 
    "尽管", "何况", "不会", "何以", "怎样", "为何", "此外", "其中","怎么","什么","为什么","是否",'。', '？', '！', '.', '?', '!','，',','
])
    
    
def hybrid_search_rrf(keyword_hits, vector_hits, k=60):
    """使用 RRF 融合两路检索结果。

    RRF 原理：每个来源的第 rank 位文档贡献 1/(k+rank) 分数，k 为平滑常数。
    该方法简单高效，能稳定融合不同来源的排序结果。
    """
    # Initialize score dictionary
    scores = {}
    
    # Process keyword hits
    for hit in keyword_hits:
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {
                'score': 0, 'text': hit['text'], 'id': doc_id,
                'file_name': hit.get('file_name'), 'page': hit.get('page'),
                'doc_type': hit.get('doc_type'), 'image_id': hit.get('image_id'),
                'table_id': hit.get('table_id')
            }
        scores[doc_id]['score'] += 1 / (k + hit['rank'])
    
    # Process vector hits
    for hit in vector_hits:
        doc_id = hit['id']
        if doc_id not in scores:
            scores[doc_id] = {
                'score': 0, 'text': hit['text'], 'id': doc_id,
                'file_name': hit.get('file_name'), 'page': hit.get('page'),
                'doc_type': hit.get('doc_type'), 'image_id': hit.get('image_id'),
                'table_id': hit.get('table_id')
            }
        scores[doc_id]['score'] += 1 / (k + hit['rank'])
    
    # Sort documents by their RRF score and assign ranks
    ranked_docs = sorted(scores.values(), key=lambda x: x['score'], reverse=True)

    # Removing the timestamps
    for _, doc in enumerate(ranked_docs):
        timestamp_pattern = re.compile(r'\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}')
        doc['text'] = re.sub(timestamp_pattern, '', doc['text'])    
        
    # Format the final list of results
    final_results = [{
        'id': doc['id'], 'text': doc['text'], 'rank': idx + 1, 'score': doc.get('score'),
        'file_name': doc.get('file_name'), 'page': doc.get('page'),
        'doc_type': doc.get('doc_type'), 'image_id': doc.get('image_id'),
        'table_id': doc.get('table_id')
    } for idx, doc in enumerate(ranked_docs)]
    # print(final_results)
    return final_results

def rerank(query, result_doc):
    """调用重排序服务（优先本地 Qwen3-Reranker，否则使用远程服务）。

    参数：
    - query: 查询文本
    - result_doc: 文档列表，每个元素含 'text' 字段

    返回：
    - 按重排序分数降序排列的文档列表
    """
    # 尝试使用本地 reranker
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        return _qwen_rerank(query, result_doc)
    except ImportError:
        # 回退到远程服务
        return _remote_rerank(query, result_doc)

def _qwen_rerank(query, result_doc):
    """使用本地 Qwen3-Reranker-0.6B 进行重排序"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import os
    
    # 懒加载模型（全局变量）
    global _reranker_model, _reranker_tokenizer
    if '_reranker_model' not in globals() or _reranker_model is None:
        print("[Loading] Qwen3-Reranker-0.6B model...")
        # 设置确定性模式（确保结果可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = '0'
        
        model_name = "Qwen/Qwen3-Reranker-0.6B"
        _reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _reranker_model.eval()
        if torch.cuda.is_available():
            _reranker_model = _reranker_model.cuda()
        print("[Loaded] Qwen3-Reranker-0.6B ready")
    
    scores = []
    with torch.no_grad():
        # 批量处理以提高效率和稳定性（如果可能）
        for doc in result_doc:
            # 构建 query-document 对
            pairs = [[query, doc.get('text', '')]]
            inputs = _reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = _reranker_model(**inputs)
            # 获取相关性分数（通常是 logits 的第一个值）
            score = outputs.logits[0][0].item() if hasattr(outputs, 'logits') else outputs[0][0].item()
            scores.append(score)
    
    # 保存原始分数用于调试（可选）
    if len(scores) > 0:
        print(f"[Rerank] Score range: {min(scores):.4f} ~ {max(scores):.4f}")
    
    # 更新文档分数并排序
    for idx, doc in enumerate(result_doc):
        result_doc[idx]['score'] = scores[idx]
    # 稳定排序：按分数降序，相同分数时按 id 排序（确保每次结果一致）
    result_doc.sort(key=lambda x: (-x.get('score', 0), x.get('id', '')))
    # 更新 rank 字段以反映新的排序
    for idx, doc in enumerate(result_doc):
        result_doc[idx]['rank'] = idx + 1
    return result_doc

def _remote_rerank(query, result_doc):
    """使用远程 rerank 服务（备用方案）"""
    res = requests.post(RERANK_URL, json={"query": query, "documents": [doc['text'] for doc in result_doc]}).json()
    if res and 'scores' in res and len(res['scores']) == len(result_doc):
        for idx, doc in enumerate(result_doc):
            result_doc[idx]['score'] = res['scores'][idx]
        # 稳定排序：按分数降序，相同分数时按 id 排序（确保每次结果一致）
        result_doc.sort(key=lambda x: (-x.get('score', 0), x.get('id', '')))
        # 更新 rank 字段以反映新的排序
        for idx, doc in enumerate(result_doc):
            result_doc[idx]['rank'] = idx + 1
    return result_doc

def rag_fusion(query):
    """将单条查询改写为多条等价或互补查询，以提升召回覆盖。"""
    prompt = f'''请根据用户的查询，将其重新改写为 2 个不同的查询。这些改写后的查询应当尽可能覆盖原始查询中的不同方面或角度，以便更全面地获取相关信息。请确保每个改写后的查询仍然与原始查询相关，并且在内容上有所不同。

用JSON的格式输出：
{{
    "rag_fusion":["query1","query2"]
}}

原始查询：{query}
'''
    # Call OpenAI ChatGPT 4o nano to generate query variations
    
    client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "你是一个智能AI助手，专注于改写用户查询，并以 JSON 格式输出"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        parsed_result = json.loads(result)
        return parsed_result.get("rag_fusion", [])
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []

def coreference_resolution(query,chat_history):
    """基于对话历史做指代消解，将含有代词的当前问题替换为明确对象的完整问题。"""
    prompt = f'''目标：根据提供的用户与知识库助手的历史记录，做指代消解，将用户最新问题中出现的代词或指代内容替换为历史记录中的明确对象，生成一条完整的独立问题。

说明：
- 将用户问题中的指代词替换为历史记录中的具体内容，生成一条独立问题。

以JSON的格式输出
{{"query":["替换指代后的完整问题"]}}

以下是一些案例

----------
历史记录：
['user': Milvus是什么?
'assistant': Milvus 是一个向量数据库]
用户问题：怎么使用它？

输出JSON：{{"query":["怎么使用Milvus?"]}}
----------
历史记录：
['user': PyTorch是什么?
'assistant': PyTorch是一个开源的机器学习库，用于Python。它提供了一个灵活且高效的框架，用于构建和训练深度神经网络。
'user': TensorFlow是什么?
'assistant': TensorFlow是一个开源的机器学习框架。它提供了一套全面的工具、库和资源，用于构建和部署机器学习模型。]
用户问题: 它们的区别是什么？

输出JSON：{{"query":["PyTorch和TensorFlow的区别是什么？"]}}
----------
历史记录：
['user': 四川有哪些城市
'assistant': 1. 成都。 2. 绵阳。 3. 资阳。]
用户问题: 介绍一下第二个

输出JSON：{{"query":["介绍一下绵阳"]}}
----------
历史记录：
{chat_history}
用户问题：{query}

输出JSON：
''' 
    # Call OpenAI ChatGPT 4o nano to generate query variations
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "你是一个智能AI助手，专注于做指代消解，并以 JSON 格式输出"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    parsed_result = json.loads(result)
    return parsed_result.get("query")


def query_decompositon(query):
    """判断是否需要将复杂查询拆分为若干子问题，提高检索与推理效果。"""
    prompt=f''' 
目标：分析用户的问题，判断其是否需要拆分为子问题以提高信息检索的准确性。如果需要拆分，提供拆分后的子问题列表；如果不需要，直接返回原问题。

说明：
- 用户的问题可能含糊不清或包含多个概念，导致难以直接回答。
- 为提高知识库查询的质量和相关性，需评估问题是否应分解为更具体的子问题。
- 根据问题的复杂性和广泛性，判断是否需要拆分：
  - 如果问题涉及多个方面（如比较多个实体、包含多个独立步骤），需要拆分为子问题。
  - 如果问题已集中且明确，无需拆分。
- 输出结果必须为 JSON 格式。请直接输出JSON，不需要做任何解释。

输出格式：
{{
  "query": ["子问题1", "子问题2"...] 
}}  

案例 1
---
用户问题: "林冲、关羽、孙悟空的性格有什么不同？"
推理过程: 该问题涉及多个实体的比较，需要分别了解每个实体的性格。
输出:
{{
  "query": ["林冲的性格是什么？", "关羽的性格是什么？", "孙悟空的性格是什么？"]
}}

案例 2
---
用户问题: "哪些OpenAI的前在职员工成立了自己的公司？"
推理过程: 解答需要先识别前员工，再判断谁创立公司，涉及多个步骤。
输出:
{{
  "query": ["OpenAI的前在职员工有哪些？", "谁成立了自己的公司？"]
}}

案例 3
---
用户问题: "Find environmentally friendly electric cars with over 300 miles of range under $40,000."
推理过程: 问题包含多个条件要求，需要拆分为具体的子问题以提高检索准确性。
输出:
{{
  "query": ["Which cars are environmentally friendly electric vehicles?", "Which electric vehicles have a range of over 300 miles?", "What electric vehicles are priced under $40,000?"]
}}

案例 4
---
用户问题: "如何设计一个智能家居系统并实时监控设备状态？"
推理过程: 问题包含两个独立方面（设计系统和监控状态），需要拆分。
输出:
{{
  "query": ["如何设计一个智能家居系统？", "如何实时监控智能家居系统的设备状态？"]
}}

案例 5
---
用户问题: "Covid对经济的影响是什么？"
推理过程: 问题集中且明确，无需拆分。
输出:
{{
  "query": []
}}

案例 6
---
用户问题: "LangChain和LangGraph的区别是什么？"
推理过程: 该问题涉及比较，可拆分为各自定义再加比较，以提高检索准确性。
输出:
{{
  "query": ["LangChain是什么？", "LangGraph是什么？"]
}}

用户问题:
"{query}"
'''
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "你是一个智能AI助手，专注于做查询拆分，并以 JSON 格式输出"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    parsed_result = json.loads(result)
    return parsed_result.get("query")

if __name__ == '__main__':
    query = '刑事诉讼法'
    # result_doc = elastic_search(query, 'test_index_1')

    # print("=== Initial Retrieval Results ===")
    # for idx, doc in enumerate(result_doc, 1):
    #     print(f"   Text: {doc['text'][:100]}{'...' if len(doc['text']) > 100 else ''}")


    # reranked_results = rerank(query, result_doc)
    # print("\n=== Reranked Results ===")
    # for idx, doc in enumerate(reranked_results, 1):
    #     print(f"   Text: {doc['text'][:100]}{'...' if len(doc['text']) > 100 else ''}")
    #     print(f"   Rerank Score: {doc.get('score', '-')}")
    #     print("-" * 60)
    
    # print(rag_fusion(query))
    
    # 多轮对话场景：用户询问关于公司法的问题，然后询问相关细节
    chat_history = """
'user': 高血压的症状有哪些？
'assistant': 高血压的常见症状包括头痛、头晕、心悸、胸闷、疲劳等。早期可能没有明显症状，但随着病情发展可能出现视力模糊、耳鸣等表现。
'user': 它的治疗方法是什么？
"""
    # 测试指代消解：用户问"它还有哪些重要条款？"，这里的"它"应该指代"民法典"
    query = "它的治疗方法是什么？"
    print("=== 指代消解测试 ===")
    print(f"原始查询: {query}")
    print(f"聊天历史: {chat_history}")
    print(f"消解结果: {coreference_resolution(query, chat_history)}")


    # query = "Find affordable smartphones with 5G connectivity, over 4000mAh battery, and excellent camera quality under $500."
    # print(f"原始查询: {query}")
    # print(f"拆分结果: {query_decompositon(query)}")
    pass