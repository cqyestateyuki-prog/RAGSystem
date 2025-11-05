## RAG Demo 项目说明

本项目实现了一个完整的 **Retrieval-Augmented Generation (RAG)** 系统，支持 PDF 文档的智能问答。

### 🎯 核心功能

#### 1. **文档处理与入库**
- ✅ PDF 文本提取与智能分块（chunk_size=1024, overlap=100）
- ✅ 图片 OCR 提取与上下文增强
- ✅ 表格结构化提取与描述生成
- ✅ 批量处理文件夹中的所有 PDF
- ✅ 向量化（Embedding）并存储到 Elasticsearch

#### 2. **混合检索系统**
- ✅ **关键词检索（BM25）**：精确匹配关键词
- ✅ **向量检索**：语义相似度搜索
- ✅ **RRF 融合**：Reciprocal Rank Fusion 算法融合两种检索结果
- ✅ **可选重排序**：Reranker 模型进一步优化排序（可选）

#### 3. **智能问答系统**
- ✅ LLM 生成答案（支持多种模型：通义千问、DeepSeek、Ollama 等）
- ✅ **自动引用来源**：显示文件名和页码
- ✅ **相关性检查**：检测问题与文档库的相关性，不相关时提示用户
- ✅ **去重显示**：自动去重同一文件同一页的多个文档片段
- ✅ **页码过滤**：自动过滤文本开头的页码，避免重复显示

#### 4. **高级增强能力**
- ✅ 查询改写（RAG Fusion）
- ✅ 指代消解（Coreference Resolution）
- ✅ 查询拆分（Query Decomposition）
- ✅ Web 搜索作为知识补充

### 📊 系统架构

```
用户问题
    ↓
[混合检索] ──→ BM25 关键词检索
    │         └─→ 向量语义检索
    ↓
[RRF 融合] ──→ 融合排序结果
    ↓
[可选 Rerank] ──→ 重排序优化（可选）
    ↓
[相关性检查] ──→ 检查检索结果相关性
    ↓
[LLM 生成答案] ──→ 基于检索到的文档生成答案
    ↓
[显示结果] ──→ 答案 + 引用来源（文件名+页码）
```

### 一、环境准备

#### 1. Python 依赖
```bash
pip install -r requirements.txt
```

#### 2. Elasticsearch 本地部署（简便方式）
**一键脚本（参考课件）**：
```bash
curl -fsSL https://elastic.co/start-local | sh
```

**启动 Elasticsearch**：
```bash
cd ~/elastic-start-local
./start.sh
```

**获取用户名密码**：
```bash
cat elastic-start-local/.env | grep ES_LOCAL_PASSWORD
cat elastic-start-local/.env | grep ES_LOCAL_USERNAME
```

**测试连接**：
```bash
curl http://elastic:ZXKwLNQD@localhost:9200
```


#### 3. LLM 配置（推荐：通义千问）

**方式1：自动配置（推荐）**
```bash
# 系统会自动加载 .env_qwen 文件（如果存在）
# 配置文件已创建：.env_qwen
# 直接启动即可，无需手动设置环境变量
python rag_system.py --index test_index_1
```

**方式2：手动设置环境变量**
```bash
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=你的_qwen_api_key
```

#### 4. 其他环境变量（可选：Web 搜索 & DeepSeek）
新建 `.env` 文件（与 `websearch.py` 一致）：
```env
WEB_SEARCH_KEY=你的_bocha_key
DEEPSEEK_API_KEY=你的_deepseek_key
```

#### 5. 外部服务地址
外部服务地址配置在 `config.py` 中：
- `EMBEDDING_URL`：向量化服务地址
- `RERANK_URL`：重排序服务地址
- `IMAGE_MODEL_URL`：多模态图片模型地址

可按需修改这些配置。

### 二、索引创建

使用 `es_functions.py` 创建向量索引：
```bash
python -c "from es_functions import create_elastic_index; create_elastic_index('test_index_1')"
```

### 三、文档处理与入库

#### 基本用法

**方式1：使用 `document_process.py`**（观察切分效果）：
```bash
python -c "from document_process import process_pdf; process_pdf('test_index_1','test_pdf/刑事诉讼法.pdf')"
```
**说明**：示例中写入 ES 的代码被注释，如需入库，取消对应注释（批量 embedding 与 `es.index` 部分）。

**方式2：使用 `run.py`（推荐，自动入库）**：
```bash
python run.py --index test_index_1 --pdf test_pdf/刑事诉讼法.pdf
```

#### 索引与文档的关系

**一个索引可以存多本 PDF**：
- 一个索引可以存储多个文档的多个分块，不是一对一关系
- 可以把多个 PDF 都写入同一个索引（便于统一检索）
- 也可以按项目/语料类型分索引（如 `laws_cn_v1`、`startup_notes_v1`）

**使用建议**：
- **统一检索**：多本 PDF 使用相同索引名，重复调用入库即可
- **数据隔离**：不同项目使用不同索引名

#### 入库示例

**入库另一本 PDF（同索引）**：
```bash
python run.py --index test_index_1 --pdf test_pdf/另一份.pdf
```

**一键创建索引 + 入库 + 检索**：
```bash
python run.py --index test_index_1 --pdf test_pdf/刑事诉讼法.pdf --query "基本原则" --top_k 3
```

#### 同一个 PDF 的完整入库流程（文本 + 图片 + 表格）

```bash
# 1. 文本入库
python run.py --index test_index_1 --pdf test_pdf/你的PDF.pdf

# 2. 图片入库（如果有）
python -c "from ingest_images_tables import ingest_images; ingest_images('test_index_1', 'test_pdf/你的PDF.pdf')"

# 3. 表格入库（如果有）
python -c "from ingest_images_tables import ingest_tables; ingest_tables('test_index_1', 'test_pdf/你的PDF.pdf')"
```

#### 批量入库（推荐：处理整个文件夹）

**批量入库文件夹中的所有 PDF**（自动处理文本、图片、表格）：
```bash
# 批量入库整个文件夹
python batch_ingest.py --index test_index_1 --pdf test_pdf/

# 只入库单个文件（也支持）
python batch_ingest.py --index test_index_1 --pdf test_pdf/刑事诉讼法.pdf
```

**选项说明**：
```bash
# 跳过图片处理（加快速度）
python batch_ingest.py --index test_index_1 --pdf test_pdf/ --no-images

# 跳过表格处理
python batch_ingest.py --index test_index_1 --pdf test_pdf/ --no-tables

# 使用本地模型
python batch_ingest.py --index test_index_1 --pdf test_pdf/ --use-local
```

**功能特点**：
- ✅ 自动扫描文件夹中的所有 PDF 文件
- ✅ 自动处理文本、图片、表格（可配置）
- ✅ 显示处理进度和统计信息
- ✅ 单个文件失败不影响其他文件

#### 手动入库多个 PDF（旧方式，不推荐）

```bash
# PDF1（只有文本）
python run.py --index test_index_1 --pdf test_pdf/刑事诉讼法.pdf

# PDF2（有图片和表格）
python run.py --index test_index_1 --pdf test_pdf/另一个PDF.pdf
python -c "from ingest_images_tables import ingest_images; ingest_images('test_index_1', 'test_pdf/另一个PDF.pdf')"
python -c "from ingest_images_tables import ingest_tables; ingest_tables('test_index_1', 'test_pdf/另一个PDF.pdf')"
```

### 四、RAG 系统启动（推荐：统一入口）

**启动 RAG 系统主界面**（推荐方式）：
```bash
python rag_system.py --index test_index_1
```

**✨ 自动配置**：系统启动时会自动加载 `.env_qwen` 配置文件（如果存在），无需手动设置环境变量！

**主界面功能**：
1. **导入 PDF 文件**：批量入库文件夹中的所有 PDF
2. **查看已导入的文档**：显示索引统计和所有参考 PDF
3. **开始问答**：启动交互式问答系统
4. **快速检索测试**：快速测试检索功能
5. **退出**

**启动时自动导入 PDF**：
```bash
# 启动时自动导入 test_pdf/ 文件夹
python rag_system.py --index test_index_1 --auto-import test_pdf/
```

**选项**：
```bash
# 使用 rerank（问答时）
python rag_system.py --index test_index_1 --rerank

# 使用本地 embedding 模型
python rag_system.py --index test_index_1 --use-local
```

### 五、分步启动（可选）

如果你更喜欢分步操作，也可以单独使用：

**批量入库**：
```bash
python batch_ingest.py --index test_index_1 --pdf test_pdf/
```

**交互式问答**：
```bash
python interactive_qa.py --index test_index_1
```

**详细启动指南**：见 `START_RAG.md`

**功能特点**：
- ✅ **持续提问**：无需重复启动，可以连续提问
- ✅ **LLM 生成答案**：使用 LLM 理解上下文并生成答案（不只是简单拼接）
- ✅ **自动引用来源**：每个答案都显示来源（文件名 + 页码）
- ✅ **智能去重**：自动去重同一文件同一页的多个文档片段
- ✅ **相关性检查**：检测问题与文档库的相关性，不相关时提示用户
- ✅ **页码过滤**：自动过滤文本开头的页码，避免重复显示
- ✅ **参考文档列表**：显示检索到的所有参考文档及其预览
- ✅ **退出命令**：输入 `exit`、`quit`、`退出` 或 `q` 退出

**问答示例**：
```
输入问题 (输入 'exit' 退出): 什么是刑事诉讼法的基本原则？

你的问题: 什么是刑事诉讼法的基本原则？
正在检索和生成答案...

============================================================
AI 回答:
============================================================
根据检索到的文档，刑事诉讼法的基本原则包括：

1. **以事实为根据，以法律为准绳**（来源: 刑事诉讼法.pdf，第3页）
2. **法律面前人人平等**（来源: 刑事诉讼法.pdf，第3页）
...

【参考文档】共 3 个:
  1. 刑事诉讼法.pdf，第3页
     以事实为根据，以法律为准绳...
  2. 刑事诉讼法.pdf，第4页
     法律面前人人平等...
  3. 刑事诉讼法.pdf，第5页
     保障人权...
```

**使用 Rerank**：
```bash
python interactive_qa.py --index test_index_1 --rerank
```

**使用本地模型**：
```bash
python interactive_qa.py --index test_index_1 --use-local
```

### 六、检索系统详解

#### 检索流程说明

系统采用**混合检索（Hybrid Search）**策略，结合关键词检索和向量检索的优势：

1. **关键词检索（BM25）**：
   - 使用 Elasticsearch 的 BM25 算法
   - 对中文查询进行分词（jieba）
   - 过滤停用词
   - 支持模糊匹配（fuzziness=AUTO）

2. **向量检索（Semantic Search）**：
   - 使用余弦相似度计算语义相似性
   - 支持中文语义理解
   - 可配置使用本地或远程 embedding 模型

3. **RRF 融合（Reciprocal Rank Fusion）**：
   - 将两种检索结果按 RRF 算法融合
   - 公式：`score = 1/(k+rank1) + 1/(k+rank2)`（k=60）
   - 兼顾精确匹配和语义匹配的优势

4. **可选重排序（Reranker）**：
   - 使用 Qwen3-Reranker-0.6B 模型
   - 进一步优化排序质量
   - 注意：可能导致排序不稳定（非确定性）

#### 查询检索测试

**✅ 方案1：使用 RRF 做最终排序（推荐，稳定且准确）**：
```bash
# RRF 融合混合检索结果（BM25 + 向量检索 + RRF 融合）
python run.py --index test_index_1 --query "未成年人刑事案件诉讼程序" --top_k 5
```
**说明**：这是默认的实现方式，`elastic_search` 函数自动使用 RRF 融合关键词和向量检索结果。

**直接调用函数**：
```python
from retrieve_documents import elastic_search
results = elastic_search('刑事诉讼法 基本原则', 'test_index_1')
for r in results[:5]:
    print(r['rank'], r['text'][:120])
```

**单行命令测试**：
```bash
python -c "from retrieve_documents import elastic_search; idx='test_index_1'; q='未成年人刑事案件诉讼程序'; res=elastic_search(q, idx); [print(f'#{i+1}: {r['text'][:160].replace('\\n',' ')}') for i,r in enumerate(res[:5])]"
```

**说明**：
- `elastic_search` 执行混合检索（BM25 + 向量 + RRF 融合）
- 结果稳定，每次运行排序一致
- 这是**推荐的使用方式**，通常已足够准确

### 七、方案2：使用 Reranker 做最终排序（可选）

**✅ 方案2：使用 Reranker Model 做最终排序**：
```bash
# 在 RRF 结果基础上，使用 Reranker 模型进一步优化排序
python run.py --index test_index_1 --query "第一章 任务和基本原则" --top_k 5 --rerank
```
**说明**：这是作业要求的另一种实现方式，使用 Qwen3-Reranker 模型对检索结果进行重排序。

**⚠️ 注意**：rerank 可能导致排序不稳定（远程服务或模型推理的非确定性），建议优先使用 RRF。

**使用方式**：
```bash
# 命令行方式（添加 --rerank 参数）
python run.py --index test_index_1 --query "第一章 任务和基本原则" --top_k 5 --rerank

# 代码方式
python -c "from retrieve_documents import elastic_search, rerank; idx='test_index_1'; q='第一章 任务和基本原则'; res=elastic_search(q, idx); res=rerank(q, res[:20]); [print(f'#{i+1}: {r['text'][:160].replace('\\n',' ')}') for i,r in enumerate(res[:5])]"
```

**为什么可能不稳定**：
- 远程 rerank 服务可能有非确定性
- 本地模型推理可能有微小差异
- 建议：**优先使用 RRF 混合检索（不加 --rerank）**，通常已足够准确且稳定

**Python 代码示例**：
```python
from retrieve_documents import elastic_search, rerank

# 初步检索
results = elastic_search('刑事诉讼法 基本原则', 'test_index_1')

# 重排序
reranked = rerank('刑事诉讼法 基本原则', results[:10])
```

### 八、高级增强能力

- 查询改写（RAG Fusion）：
```python
from retrieve_documents import rag_fusion
print(rag_fusion('刑事诉讼法的基本原则是什么？'))
```

- 指代消解：
```python
from retrieve_documents import coreference_resolution
chat_history = """
'user': 什么是刑事诉讼法？
'assistant': 刑事诉讼法是规范刑事诉讼程序的法律…
"""
print(coreference_resolution('它的基本原则是什么？', chat_history))
```

- 查询拆分：
```python
from retrieve_documents import query_decompositon
print(query_decompositon('Find EVs >300 miles under $40k and eco-friendly'))
```

### 九、多模态：PDF 图片/表格

#### 测试图片和表格提取（仅提取，不入库）

**测试图片提取**：
```bash
python -c "from image_table import extract_images_from_pdf; extract_images_from_pdf('test_pdf/image_extraction_example.pdf')"
```
**说明**：会显示图片的原始描述和上下文增强后的描述。

**测试表格提取**：
```bash
python -c "from image_table import extract_tables_from_pdf; extract_tables_from_pdf('test_pdf/table_extraction_example.pdf')"
```
**说明**：会显示表格的 Markdown 格式和上下文增强后的说明。

#### 图片和表格入库（完整流程）

**图片入库**：
```bash
python -c "from ingest_images_tables import ingest_images; ingest_images('test_index_1', 'test_pdf/image_extraction_example.pdf')"
```

**表格入库**：
```bash
python -c "from ingest_images_tables import ingest_tables; ingest_tables('test_index_1', 'test_pdf/table_extraction_example.pdf')"
```

**检索测试**（验证是否能检索到图片/表格的描述）：
```bash
# 检索图片描述
python run.py --index test_index_1 --query "图片中显示的内容" --top_k 5

# 检索表格描述
python run.py --index test_index_1 --query "表格中的数据" --top_k 5
```

**查看检索结果的类型**：
```bash
python -c "from retrieve_documents import elastic_search; res=elastic_search('图片内容', 'test_index_1'); [print(f'doc_type={r.get(\"doc_type\")}, page={r.get(\"page\")}, image_id={r.get(\"image_id\")}') for r in res[:5]]"
```

**注意**：
- `summarize_image` 需要可用的多模态模型服务（`IMAGE_MODEL_URL`）
- 确保 `image_table.py` 中填入可用的 `api_key` 或由网关托管
- 详细测试步骤见 `test_images_tables.md`

### 十、Web 搜索作为 RAG

```python
from websearch import bocha_web_search, ask_llm
query = 'RAG Fusion 的优点'
webctx = bocha_web_search(query)
print(ask_llm(query, webctx))
```

### 十一、代码结构说明

#### 核心文件说明

| 文件 | 功能说明 |
|------|---------|
| `rag_system.py` | **主入口**：统一启动界面，提供菜单选项 |
| `interactive_qa.py` | **交互式问答**：LLM 生成答案，显示引用来源 |
| `batch_ingest.py` | **批量入库**：扫描文件夹，自动处理文本/图片/表格 |
| `retrieve_documents.py` | **检索核心**：混合检索、RRF 融合、Rerank |
| `document_process.py` | **文档处理**：PDF 文本提取与分块 |
| `image_table.py` | **多模态处理**：图片/表格提取与上下文增强 |
| `embedding.py` | **向量化**：Embedding 模型调用（本地/远程） |
| `es_functions.py` | **Elasticsearch**：索引创建与管理 |
| `config.py` | **配置管理**：连接配置、环境变量 |
| `run.py` | **快速测试**：单文件入库与检索测试 |

#### 工作流程

```
1. 文档入库流程：
   batch_ingest.py / run.py
      ↓
   document_process.py (文本分块)
      ↓
   embedding.py (向量化)
      ↓
   es_functions.py (写入 ES)

2. 问答流程：
   interactive_qa.py
      ↓
   retrieve_documents.py (混合检索 + RRF)
      ↓
   [可选] rerank (重排序)
      ↓
   LLM 生成答案 (openai 兼容接口)
      ↓
   显示答案 + 引用来源
```

#### 关键技术点

- **混合检索**：BM25 + 向量检索，兼顾精确匹配和语义匹配
- **RRF 融合**：稳定且高效的排序融合算法
- **元数据管理**：文件名、页码、文档类型等元数据完整存储
- **智能去重**：按文件+页码去重，避免重复显示
- **相关性检查**：基于 RRF 分数判断相关性，提升答案质量

### 十二、常见问题（FAQ）

**Q: Elasticsearch 连接失败？**
- 检查 `config.py` 的 `ElasticConfig.url` 与本地 ES 是否运行
- 若使用上面的 quickstart，请根据脚本创建的 `.env` 更新账号密码

**Q: 检索结果为空？**
- 确认已入库（使用 `run.py --pdf` 或取消 `document_process.py` 中写入 ES 注释后再执行）
- 确认索引映射的向量维度与 embedding 服务一致（默认 1024）

**Q: 依赖缺失告警？**
- 运行 `pip install -r requirements.txt`
- 必要时激活正确的虚拟环境

**Q: 向量模型一直下载？**
- 默认使用远程服务（`use_local=False`），无需下载模型
- 如需使用本地模型，添加 `--use-local` 参数（首次会下载模型）

**Q: Rerank 排序不稳定？**
- 这是正常现象，rerank 可能因远程服务或模型推理的非确定性导致
- 建议优先使用 RRF 混合检索（不加 `--rerank`），通常已足够准确且稳定

**Q: 参考文档数量少于 5 条？**
- 系统会自动去重同一文件同一页的多个文档片段
- 如果检索结果包含多个来自同一页的片段，去重后显示数量会减少
- 这是正常现象，确保每个来源只显示一次

**Q: 问题与文档库不相关时如何处理？**
- 系统会自动检查检索结果的相关性分数
- 如果相关性太低，会提示用户并建议使用更具体的关键词
- LLM 也会被告知不要编造答案，明确说明无法回答

**Q: 答案中没有引用来源？**
- 确保 LLM 配置正确（检查 `.env_qwen` 或环境变量）
- 系统会要求 LLM 在答案中标注来源（文件名+页码）
- 如果 LLM 未按要求标注，可以在答案下方的参考文档列表中查看来源


