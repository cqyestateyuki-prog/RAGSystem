# 图片和表格测试指南

## 一、仅提取测试（查看效果，不入库）

### 1. 测试图片提取

**提取图片并生成描述**：
```bash
python -c "from image_table import extract_images_from_pdf; extract_images_from_pdf('test_pdf/image_extraction_example.pdf')"
```

**预期输出**：
- 会显示每张图片的：
  - 原始描述（多模态模型生成）
  - 上下文增强后的描述（结合页面上下文）
  - 图片保存路径（临时保存，处理完后会删除）

### 2. 测试表格提取

**提取表格并生成描述**：
```bash
python -c "from image_table import extract_tables_from_pdf; extract_tables_from_pdf('test_pdf/table_extraction_example.pdf')"
```

**预期输出**：
- 会显示每个表格的：
  - Markdown 格式的表格内容
  - 页面上下文
  - 上下文增强后的表格说明（3句话以内的描述）

## 二、完整测试（提取+入库+检索）

### 步骤1：确保索引已创建
```bash
python -c "from es_functions import create_elastic_index; create_elastic_index('test_index_1')"
```

### 步骤2：图片入库
```bash
python -c "from ingest_images_tables import ingest_images; ingest_images('test_index_1', 'test_pdf/image_extraction_example.pdf')"
```

**预期输出**：
```
[Images] Ingested X image descriptions into test_index_1
```

### 步骤3：表格入库
```bash
python -c "from ingest_images_tables import ingest_tables; ingest_tables('test_index_1', 'test_pdf/table_extraction_example.pdf')"
```

**预期输出**：
```
[Tables] Ingested X table descriptions into test_index_1
```

### 步骤4：检索测试（验证是否能检索到）

**检索图片描述**：
```bash
python run.py --index test_index_1 --query "图片中显示的内容" --top_k 5
```

**检索表格描述**：
```bash
python run.py --index test_index_1 --query "表格中的数据" --top_k 5
```

**查看检索结果的元数据**：
```bash
python -c "from retrieve_documents import elastic_search; res=elastic_search('图片内容', 'test_index_1'); [print(f'#{i+1}: doc_type={r.get(\"doc_type\")}, page={r.get(\"page\")}, image_id={r.get(\"image_id\")}, text={r[\"text\"][:100]}') for i,r in enumerate(res[:5])]"
```

## 三、一键测试脚本

创建一个测试脚本 `test_images_tables.py`：

```python
#!/usr/bin/env python3
"""一键测试图片和表格的提取、入库和检索"""

from ingest_images_tables import ingest_images, ingest_tables
from retrieve_documents import elastic_search
from es_functions import create_elastic_index

INDEX_NAME = "test_index_1"

def test_images():
    print("=" * 60)
    print("测试图片提取和入库")
    print("=" * 60)
    
    # 入库图片
    ingest_images(INDEX_NAME, "test_pdf/image_extraction_example.pdf")
    
    # 检索测试
    print("\n检索图片描述：")
    results = elastic_search("图片中显示的内容", INDEX_NAME)
    image_results = [r for r in results[:5] if r.get("doc_type") == "image"]
    for r in image_results:
        print(f"  [{r.get('file_name')} - p.{r.get('page')}] {r['text'][:100]}...")

def test_tables():
    print("=" * 60)
    print("测试表格提取和入库")
    print("=" * 60)
    
    # 入库表格
    ingest_tables(INDEX_NAME, "test_pdf/table_extraction_example.pdf")
    
    # 检索测试
    print("\n检索表格描述：")
    results = elastic_search("表格中的数据", INDEX_NAME)
    table_results = [r for r in results[:5] if r.get("doc_type") == "table"]
    for r in table_results:
        print(f"  [{r.get('file_name')} - p.{r.get('page')}] {r['text'][:100]}...")

if __name__ == "__main__":
    # 确保索引存在
    create_elastic_index(INDEX_NAME)
    
    # 测试图片
    test_images()
    
    # 测试表格
    test_tables()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
```

**运行测试脚本**：
```bash
python test_images_tables.py
```

## 四、验证结果

### 检查 ES 中的数据

**查看图片数据**：
```bash
python -c "from config import get_es; es=get_es(); res=es.search(index='test_index_1', query={'term': {'doc_type': 'image'}}, size=5); print(f'找到 {res[\"hits\"][\"total\"][\"value\"]} 条图片记录'); [print(f'  - {h[\"_source\"][\"file_name\"]} p.{h[\"_source\"][\"page\"]}') for h in res['hits']['hits']]"
```

**查看表格数据**：
```bash
python -c "from config import get_es; es=get_es(); res=es.search(index='test_index_1', query={'term': {'doc_type': 'table'}}, size=5); print(f'找到 {res[\"hits\"][\"total\"][\"value\"]} 条表格记录'); [print(f'  - {h[\"_source\"][\"file_name\"]} p.{h[\"_source\"][\"page\"]}') for h in res['hits']['hits']]"
```

## 五、注意事项

1. **多模态模型服务**：图片描述需要 `IMAGE_MODEL_URL` 可用，检查 `config.py` 和 `image_table.py` 中的配置
2. **API Key**：确保 `image_table.py` 中的 `YOUR_API_KEY` 已替换为有效的 API Key
3. **处理时间**：图片和表格处理需要调用 LLM，可能较慢
4. **文件路径**：确保测试 PDF 文件存在（`test_pdf/image_extraction_example.pdf` 和 `test_pdf/table_extraction_example.pdf`）

