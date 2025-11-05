#!/bin/bash
# 通义千问快速配置脚本

echo "配置通义千问 API Key..."

# 设置环境变量（当前终端有效）
# 通义千问的兼容模式地址（正确地址）
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=sk-f8b94bf3cdb24bddaa40a3cf9c4c500c

echo "✅ 环境变量已设置（当前终端有效）"
echo ""
echo "API Base URL: $OPENAI_API_BASE_URL"
echo "API Key: ${OPENAI_API_KEY:0:10}...（已隐藏）"
echo ""
echo "测试连接中..."
python3 -c "
from openai import OpenAI
import os
try:
    client = OpenAI(
        base_url=os.getenv('OPENAI_API_BASE_URL'),
        api_key=os.getenv('OPENAI_API_KEY')
    )
    response = client.chat.completions.create(
        model='qwen-plus',
        messages=[{'role':'user','content':'你好'}],
        max_tokens=50
    )
    print('✅ 连接成功！')
    print('回答:', response.choices[0].message.content)
except Exception as e:
    print('❌ 连接失败:', str(e)[:200])
    print('')
    print('可能的原因：')
    print('1. API Key 无效或已过期')
    print('2. API 地址不正确')
    print('3. 网络问题')
    print('')
    print('请检查：')
    print('- API Key 是否正确（从 https://dashscope.console.aliyun.com/ 获取）')
    print('- 是否已开通通义千问服务')
"

echo ""
echo "如果测试成功，运行以下命令启动 RAG 系统："
echo "  python rag_system.py --index test_index_1"
echo ""
echo "注意：这些环境变量只在当前终端有效，关闭终端后需要重新设置"
echo "或添加到 ~/.zshrc 或 ~/.bashrc 中永久保存"

