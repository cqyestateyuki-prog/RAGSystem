"""
Embedding 服务封装：
- 支持本地 Qwen3-Embedding-0.6B 模型（推荐）
- 也支持远程向量服务（如果配置了 EMBEDDING_URL）
- 向量维度：1024（与 ES 索引配置一致）
"""

import os
from typing import List
from config import EMBEDDING_URL
import requests

# 尝试导入 transformers，如果失败则使用远程服务
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    HAS_LOCAL_MODEL = True
except ImportError:
    HAS_LOCAL_MODEL = False

# 全局模型变量（懒加载）
_embedding_model = None
_embedding_tokenizer = None

def _load_qwen_embedding_model():
    """懒加载 Qwen3-Embedding 模型（首次调用时加载）"""
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None:
        print("[Loading] Qwen3-Embedding-0.6B model...")
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        _embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _embedding_model = AutoModel.from_pretrained(model_name)
        _embedding_model.eval()  # 设置为评估模式
        if torch.cuda.is_available():
            _embedding_model = _embedding_model.cuda()
        print("[Loaded] Qwen3-Embedding-0.6B ready")
    return _embedding_model, _embedding_tokenizer

def local_embedding(inputs: List[str], use_local: bool = False) -> List[List[float]]:
    """生成文本向量（优先使用本地 Qwen3 模型）。

    参数：
    - inputs: 列表[str]，每个元素为待编码文本
    - use_local: 是否使用本地模型（默认 False，使用远程服务）

    返回：
    - List[List[float]]，与 inputs 等长的向量列表，每个向量 1024 维
    """
    if use_local and HAS_LOCAL_MODEL:
        return _qwen_embedding(inputs)
    else:
        # 回退到远程服务
        return _remote_embedding(inputs)

def _qwen_embedding(inputs: List[str]) -> List[List[float]]:
    """使用本地 Qwen3-Embedding-0.6B 生成向量（1024维）"""
    model, tokenizer = _load_qwen_embedding_model()
    embeddings = []
    
    with torch.no_grad():
        # 批量处理以提高效率
        texts = inputs
        # Tokenize 所有文本
        inputs_tokenized = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        if torch.cuda.is_available():
            inputs_tokenized = {k: v.cuda() for k, v in inputs_tokenized.items()}
            model = model.cuda()
        
        # 生成 embedding
        outputs = model(**inputs_tokenized)
        
        # 使用 mean pooling（对序列维度取平均）
        # outputs.last_hidden_state 形状: [batch_size, seq_len, hidden_size]
        # mean(dim=1) 后: [batch_size, hidden_size]
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # 转换为 CPU numpy 再转 Python list
        for emb in batch_embeddings:
            embedding = emb.cpu().numpy().tolist()
            # Qwen3-Embedding-0.6B 默认输出应该是 1024 维
            # 如果实际维度不是 1024，需要调整 ES 索引配置或模型输出
            embeddings.append(embedding)
    
    return embeddings

def _remote_embedding(inputs: List[str]) -> List[List[float]]:
    """使用远程 embedding 服务（备用方案）"""
    headers = {"Content-Type": "application/json"}
    data = {"texts": inputs}
    # 打印日志确认使用远程服务（首次调用时）
    if not hasattr(_remote_embedding, '_logged'):
        print(f"[Embedding] Using remote service: {EMBEDDING_URL}")
        _remote_embedding._logged = True
    
    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # 如果状态码不是 200，会抛出异常
        
        # 检查响应内容是否为空
        if not response.text:
            raise ValueError(f"远程 embedding 服务返回空响应: {EMBEDDING_URL}")
        
        # 尝试解析 JSON
        try:
            result = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            # 如果解析失败，打印响应内容以便调试
            print(f"[错误] 远程 embedding 服务返回无效的 JSON")
            print(f"[调试] 响应状态码: {response.status_code}")
            print(f"[调试] 响应内容前 500 字符: {response.text[:500]}")
            raise ValueError(f"远程 embedding 服务返回无效的 JSON: {e}")
        
        # 检查返回的数据结构
        if 'data' not in result or 'text_vectors' not in result['data']:
            print(f"[错误] 远程 embedding 服务返回的数据格式不正确")
            print(f"[调试] 返回的 JSON: {result}")
            raise ValueError(f"远程 embedding 服务返回的数据格式不正确，期望包含 'data.text_vectors'")
        
        return result['data']['text_vectors']
    except requests.exceptions.RequestException as e:
        print(f"[错误] 远程 embedding 服务请求失败: {e}")
        print(f"[提示] 请检查 EMBEDDING_URL 配置是否正确: {EMBEDDING_URL}")
        print(f"[提示] 或者尝试使用本地模型: --use-local")
        raise

def openai_embedding(inputs):
    """OpenAI Embedding（预留接口）"""
    pass

if __name__ == '__main__':
    inputs = ["Hello, world!"]
    output = local_embedding(inputs)[0]
    print(output)
    print("Dim: ",len(output))
    