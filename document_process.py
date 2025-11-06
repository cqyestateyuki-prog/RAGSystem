"""
PDF 文档处理模块：
1) 读取 PDF 并将其按语义合理的粒度进行分块（chunking）
2) 可批量调用向量服务生成 embedding
3) 将文本与向量写入 Elasticsearch 索引

注意：当前示例中批量写入与索引代码被注释，便于先验证切分流程。
"""

from config import get_es
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding import local_embedding
import time
import tiktoken

def process_pdf(es_index, file_path):
    """处理单个 PDF：加载 -> 分块 ->（可选）向量化 ->（可选）写入 ES。

    参数：
    - es_index: Elasticsearch 索引名
    - file_path: 本地 PDF 路径
    """
    es = get_es()
    loader = PyMuPDFLoader(file_path) # 如果报错则使用 PyMuPDFLoader 处理 pdf 文件
    pages = loader.load()

    # 使用递归分割器，保证在自然断点处切分；重叠 100 token 保留跨块上下文
    textsplit = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, length_function=num_tokens_from_string)
    chunks = textsplit.split_documents(pages)
    batch = []
    for i, chunk in enumerate(chunks): # 收集 25 个 chunks 为一批送到嵌入模型，提升吞吐
        print(chunk)
        batch.append(chunk)

        if len(batch) == 25 or i == len(chunks) - 1: 
             embeddings = local_embedding([b.page_content for b in batch])
             for j, pc in enumerate(batch):
                 body = {
                     'text': pc.page_content,
                     'vector': embeddings[j],
                 }
                 retry = 0
                 while retry <= 5:
                     try:
                         # print(body)
                         es.index(index=es_index, body=body) # 写入 elastic
                         break
                     except Exception as e:
                         print(f'[Elastic Error] {str(e)} retry')
                         retry += 1
                         time.sleep(1)

             batch = []
            
def num_tokens_from_string(string):   
    """使用 cl100k_base 编码估算 token 数，便于基于 token 粒度做分块。"""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    process_pdf('test_index_1', 'test_pdf/刑事诉讼法.pdf')