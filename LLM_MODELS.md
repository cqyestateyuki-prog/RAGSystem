# RAG 系统可用的 AI 模型选项

## 免费/开源模型选项

### 1. DeepSeek（推荐，免费且强大）

**特点**：
- ✅ 完全免费
- ✅ 性能优秀
- ✅ 支持中文
- ✅ 有 API 接口

**配置方法**：

在 `.env` 文件中添加：
```env
DEEPSEEK_API_KEY=你的_deepseek_key
```

或设置环境变量：
```bash
export DEEPSEEK_API_KEY=你的_deepseek_key
export OPENAI_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

**获取 API Key**：
- 访问：https://www.volcengine.com/product/ark
- 注册火山引擎账号
- 申请 DeepSeek API Key

**代码中模型名称**：`deepseek-v3-250324` 或 `deepseek-chat`

---

### 2. 通义千问（Qwen）

**特点**：
- ✅ 阿里云提供
- ✅ 支持中文
- ✅ 有免费额度

**配置方法**：
```bash
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=你的_qwen_key
```

**注意**：地址是 `aliyuncs.com`（不是 `aliyun.com`）

**模型名称**：`qwen-plus`, `qwen-max`, `qwen-turbo`

**获取 API Key**：
- 访问：https://dashscope.console.aliyun.com/
- 注册阿里云账号
- 申请 DashScope API Key

---

### 3. 本地模型（Ollama）

**特点**：
- ✅ 完全免费
- ✅ 本地运行，数据安全
- ✅ 无需网络

**安装 Ollama**：
```bash
# macOS
brew install ollama

# 或访问 https://ollama.ai 下载
```

**启动本地模型**：
```bash
# 下载模型（首次需要）
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

# 启动服务
ollama serve
```

**配置方法**：
```bash
export OPENAI_API_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama  # 可以是任意值
```

**模型名称**：`qwen2.5:7b`, `llama3.2:3b`, `mistral` 等

---

### 4. 其他 OpenAI 兼容接口

**很多模型服务都提供 OpenAI 兼容接口**，包括：
- **Groq**（快速，免费额度）
- **Together.ai**（多种开源模型）
- **Anthropic Claude**（需要 API Key）
- **本地部署的 vLLM**（高性能）

**配置方法**：
```bash
export OPENAI_API_BASE_URL=你的服务地址
export OPENAI_API_KEY=你的_api_key
```

---

## 快速配置指南

### 方法1：使用环境变量（推荐）

创建 `.env` 文件：
```env
# DeepSeek（推荐）
DEEPSEEK_API_KEY=你的_deepseek_key
OPENAI_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 或使用 Ollama（本地）
# OPENAI_API_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama
```

运行：
```bash
python rag_system.py --index test_index_1
```

### 方法2：修改代码中的默认配置

在 `interactive_qa.py` 中修改默认配置：
```python
# 默认使用 DeepSeek
base_url = os.getenv("OPENAI_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
```

---

## 推荐方案

### 方案1：DeepSeek（最简单）
```bash
# 1. 获取 DeepSeek API Key
# 2. 设置环境变量
export DEEPSEEK_API_KEY=你的key
export OPENAI_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 3. 启动系统
python rag_system.py --index test_index_1
```

### 方案2：本地 Ollama（最安全）
```bash
# 1. 安装并启动 Ollama
ollama pull qwen2.5:7b
ollama serve

# 2. 设置环境变量
export OPENAI_API_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama

# 3. 启动系统
python rag_system.py --index test_index_1
```

### 方案3：暂时不用 LLM（只显示检索结果）
如果不想配置 LLM，系统会自动回退到简单拼接方式，仍然可以查看检索到的文档内容。

---

## 测试模型连接

```python
# 测试脚本 test_llm.py
from openai import OpenAI
import os

base_url = os.getenv("OPENAI_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model="deepseek-v3-250324",  # 或 "qwen2.5:7b" (Ollama)
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

运行：
```bash
python test_llm.py
```

