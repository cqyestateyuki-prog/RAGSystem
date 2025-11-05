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

### 第二步：启动交互式问答系统

**启动交互式问答界面：**

```bash
python interactive_qa.py --index test_index_1
```

**启动后你会看到：**
```
============================================================
PDF 处理完成! 现在可以提问了
============================================================

[索引] test_index_1 包含 XXX 条文档

[参考文档] 共 X 个 PDF 文件：
  1. 刑事诉讼法.pdf (XXX 条文档)
  2. 刑法.pdf (XXX 条文档)
  ...

输入问题 (输入 'exit' 退出): 
```

**选项**：
```bash
# 使用 rerank（可能更精确但不稳定）
python interactive_qa.py --index test_index_1 --rerank

# 使用本地 embedding 模型
python interactive_qa.py --index test_index_1 --use-local
```

## 完整示例

```bash
# 1. 批量入库（一次性，完成后所有 PDF 都入库了）
python batch_ingest.py --index test_index_1 --pdf test_pdf/

# 2. 启动问答系统（可以持续提问）
python interactive_qa.py --index test_index_1

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

1. **首次使用**：需要先运行 `batch_ingest.py` 入库 PDF
2. **后续使用**：如果 PDF 没变化，直接运行 `interactive_qa.py` 即可
3. **更新文档**：如果添加了新 PDF，重新运行 `batch_ingest.py`（会自动跳过已存在的，或需要先删除索引）

