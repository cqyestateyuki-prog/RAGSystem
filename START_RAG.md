# RAG 系统启动指南

## 快速启动流程

### 第一步：批量入库 PDF（一次性）

**把所有参考 PDF 放到 `test_pdf/` 文件夹，然后运行：**

```bash
python batch_ingest.py --index test_index_1 --pdf test_pdf/
```

**说明**：
- 自动扫描 `test_pdf/` 文件夹中的所有 PDF（包括子文件夹）
- 自动处理文本、图片、表格
- 显示处理进度和统计信息
- 完成后会显示所有参考的 PDF 文件列表

**选项**：
```bash
# 只处理文本（跳过图片和表格，加快速度）
python batch_ingest.py --index test_index_1 --pdf test_pdf/ --no-images --no-tables

# 使用本地模型（如果配置了）
python batch_ingest.py --index test_index_1 --pdf test_pdf/ --use-local
```

### 第二步：启动 RAG 系统

**启动 RAG 系统主界面：**

```bash
python rag_system.py
```

**启动后你会看到菜单：**
```
=== PDF RAG System ===
Options:
1. Create index
2. Delete index
3. Add PDF to existing index
4. Search in existing index
5. List indices
6. Start QA in index
7. Web search QA
8. RAG Fusion QA (multi-query)
9. Exit

Enter choice (1-9):
```

**主要功能说明**：
- **选项1**：创建新的 Elasticsearch 索引
- **选项3**：批量导入 PDF 文件（文本、图片、表格）
- **选项4**：快速检索测试（不生成答案）
- **选项6**：启动交互式问答系统（推荐）
- **选项8**：RAG Fusion 多查询融合问答

**直接进入问答模式（如果已指定索引）**：
```bash
python rag_system.py --index test_index_1
```

**选项**：
```bash
# 使用 rerank（可能更精确但不稳定）
python rag_system.py --index test_index_1 --rerank

# 使用本地 embedding 模型（必须与索引创建时一致）
python rag_system.py --index test_index_1 --use-local
```

**⚠️ 重要提示：Embedding 模型一致性**
- 如果索引是用**本地模型**创建的（选项3时选择了 `y`），检索时也要选择使用本地模型
- 如果索引是用**远程服务**创建的（选项3时选择了 `n`），检索时也要选择使用远程服务
- 模型不一致会导致检索结果完全不同！

## 完整示例

### 方式1：使用菜单系统（推荐）

```bash
# 1. 启动 RAG 系统
python rag_system.py

# 2. 选择选项1：创建索引（如果还没有）
# 3. 选择选项3：添加 PDF 文件
#    - 输入索引名：test_index_1
#    - 输入 PDF 路径：test_pdf/
#    - 选择是否处理图片和表格
#    - 选择是否使用本地 embedding 模型（记住这个选择！）

# 4. 选择选项6：开始问答
#    - 输入索引名：test_index_1
#    - 选择是否使用本地 embedding 模型（必须与步骤3一致！）
#    - 开始提问
```

### 方式2：命令行直接启动

```bash
# 1. 批量入库（一次性，完成后所有 PDF 都入库了）
python batch_ingest.py --index test_index_1 --pdf test_pdf/

# 2. 启动问答系统（可以持续提问）
python rag_system.py --index test_index_1

# 3. 在问答界面中提问
# 输入问题: 什么是贝叶斯统计
# 系统会检索并生成答案，显示引用来源
```

## 其他命令（可选）

**简单检索（非交互式）**：
```bash
python run.py --index test_index_1 --query "你的问题" --top_k 5
```

**使用 rerank**：
```bash
python run.py --index test_index_1 --query "你的问题" --top_k 5 --rerank
```

## 注意事项

1. **首次使用**：需要先创建索引并导入 PDF（使用菜单选项1和3，或使用 `batch_ingest.py`）
2. **后续使用**：如果 PDF 没变化，直接运行 `rag_system.py` 选择选项6开始问答
3. **更新文档**：如果添加了新 PDF，使用菜单选项3重新导入（或使用 `batch_ingest.py`）
4. **⚠️ Embedding 模型一致性**：
   - 索引创建时使用的 embedding 模型（本地/远程）必须与检索时一致
   - 如果不一致，检索结果会完全不同
   - 建议：创建索引时记住使用的模型类型，检索时选择相同的模型

