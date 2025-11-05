"""
Web 搜索作为 RAG 的外部检索源：
- 调用博查 API 获取实时网页结果（需环境变量 WEB_SEARCH_KEY）
- 将网页内容（标题/摘要/时间）拼装为参考上下文，交给 LLM 生成答案

也支持调用字节火山引擎托管的 DeepSeek 模型（需 DEEPSEEK_API_KEY）。
"""

from copy import deepcopy
import requests,json,os

from dotenv import load_dotenv

load_dotenv()
WEB_SEARCH_KEY = os.getenv('WEB_SEARCH_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


def bocha_web_search(web_query):
    """调用博查 Web Search，将结果整理为可读的多段文本。"""
    key=WEB_SEARCH_KEY
    url = "https://api.bochaai.com/v1/web-search"

    payload = json.dumps({
    "query": web_query,
    "count": 10,
    "summary": True,
    })

    headers = {
    'Authorization': f'Bearer {key}',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.json())
    web_pages = response.json().get('data', {}).get('webPages', {}).get('value', [])

    top_results = []
    for page in web_pages:

        result = {
            'title': page.get('name'),
            'url': page.get('url'),
            'date': page.get('dateLastCrawled'),
            'source': page.get('siteName'),
            'logo': page.get('siteIcon'),
            'summary': page.get('summary'),
            'snippet': page.get('snippet')
        }
        top_results.append(result)
        
    # 组织为“标题/日期/内容”的多段文本，便于与用户问题一并喂给 LLM
    web_articles_text = '\n\n```\n'.join(
        f"标题：{web.get('title', '无标题')}\n"
        f"日期：{web.get('date', '未知日期')}\n"
        f"内容：{web.get('summary', '')}"
        for web in top_results
    )
    return web_articles_text


def ask_llm(query,websearch=None):
    """调用 LLM 生成答案；可附带 websearch 结果作为上下文提示。"""
    from openai import OpenAI
    
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=DEEPSEEK_API_KEY
    )
    
    response = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {"role": "user", "content": f"{query}\n\n参考资料：\n{websearch}" if websearch is not None else query}
        ],
        max_tokens=4000,
        timeout=60
    )
    
    return response.choices[0].message.content

if __name__ == '__main__':
    query = "查理科克遇刺"
    print(ask_llm(query))

    # print(ask_llm(query,bocha_web_search(query)))
    # print(bocha_web_search(query))