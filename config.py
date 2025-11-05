"""
全局配置：
- Elasticsearch 连接与重试封装
- 各种外部服务的基础 URL（Embedding / Rerank / 多模态图片模型）
"""

from elasticsearch import Elasticsearch
import time


class ElasticConfig:
    # 示例本地连接（包含用户名/密码），按需替换
    # 从 ~/elastic-start-local/.env 获取的密码
    url='http://elastic:ZXKwLNQD@localhost:9200'
    
    
def get_es():
    """获取 ES 客户端；若失败则每 3 秒重试，直到成功。"""
    while True:
        try:    
            es = Elasticsearch([ElasticConfig.url]) 
            return es
        except:
            print('ElasticSearch conn failed, retry')
            time.sleep(3)
            
# 向量服务 / 重排序 / 多模态模型等服务地址（示例环境）
EMBEDDING_URL="http://test.2brain.cn:9800/v1/emb"
RERANK_URL="http://test.2brain.cn:2260/rerank"
IMAGE_MODEL_URL='http://test.2brain.cn:23333/v1'
