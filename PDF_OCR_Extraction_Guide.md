## PDF 文本/图片/表格处理与 OCR 增强指南（实践版）

目标：在现有 RAG 管线（PyMuPDF + 分块 + 向量化 + ES）基础上，稳定处理三类 PDF：
- 电子 PDF（有文本层）
- 扫描/图片 PDF（无文本层，需要 OCR）
- 含图片/表格的复杂版面（需要描述/结构化）

### 一、总体策略

1) 电子 PDF：优先直接读取“文本层”（无需 OCR）
- 使用 `PyMuPDFLoader` 读取页面文本；当前项目已采用此路线，速度快、质量稳定。

2) 扫描/图片 PDF：做“页级 OCR 兜底”
- 判断当前页是否几乎没有文本（字数/tokens 接近 0）→ 对该页执行 OCR，将识别结果作为该页文本；再走分块/入库。

3) 图片与表格：最小闭环 > 完美还原
- 图片：提取大图 → 多模态描述 → 结合页面文字做“上下文增强” → 当作文本向量入库。
- 表格：检测 → 转 Markdown（保结构）→ 生成“表意说明” → 作为文本向量入库。

### 二、现有代码如何复用

- 文本：`document_process.py` / `run.py`（推荐 `run.py` 入库）
- 图片：`image_table.py::extract_images_from_pdf`
- 表格：`image_table.py::extract_tables_from_pdf`

最小命令：
```bash
# 图片描述与增强（打印结果供人工检查）
python -c "from image_table import extract_images_from_pdf; extract_images_from_pdf('test_pdf/image_extraction_example.pdf')"

# 表格抽取与增强（打印 Markdown 与说明）
python -c "from image_table import extract_tables_from_pdf; extract_tables_from_pdf('test_pdf/table_extraction_example.pdf')"
```

### 三、OCR 兜底的接入点（建议）

场景：页面文本接近为空（例如 `len(page_text.strip()) < 20`）。

可选 OCR 方案：
- Tesseract（纯本地、易装，中文需装训练数据）
- PaddleOCR（对中/英/多语更友好，依赖略多）

伪代码（接到 `run.py`/`document_process.py` 的“页面读取”处）：
```python
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

def read_page_text_with_ocr_fallback(pdf_path, page_index):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    text = page.get_text("text")
    if text and text.strip():
        return text

    # OCR 兜底：将整页渲染成图片后 OCR
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    return ocr_text
```

集成方式：
- 在分块前，用上述函数替换/补充页面文本；将 OCR 的文本与元数据（`file_name/page`）一并入库。
- OCR 仅对“文本为空”的页面触发，避免对电子 PDF 降速。

安装参考：
```bash
# macOS 示例
brew install tesseract
brew install tesseract-lang # 或手动安装中文训练数据 chi_sim.traineddata
pip install pytesseract pillow
```

### 四、图片处理要点（现有实现）

管线：提取大图 → 多模态 caption → 上下文增强 → 入库（当文本）
- 入口：`image_table.extract_images_from_pdf`（已过滤小图，做了 CMYK→RGB 转换）
- 描述：`summarize_image`（多模态模型）
- 增强：`context_augmentation`（结合页文字，补充“图意/用途/范围”）
- 入库：将增强后的描述当“文本块”向量化写入 ES（你可沿用 `run.py` 的写入逻辑）

小贴士：
- 过滤小图/装饰图（已内置宽度阈值，必要时可加自定义过滤器）
- 引用：保留 `file_name/page` 元数据，便于答案里引用“图在第几页”。

### 五、表格处理要点（现有实现）

管线：检测表格 → 转 Markdown → 生成“表意说明” → 入库（当文本）
- 入口：`image_table.extract_tables_from_pdf`
- Markdown：`page.find_tables().to_markdown()`（fitz 自带表格检测能力）
- 增强：`table_context_augmentation` 生成简短“表意说明”（3 句话内）
- 入库：Markdown + 说明均可向量化写入（建议合成一个文本块）

遇到 fitz 检不出表格：
- 备选：Camelot/Tabula（线框表格更稳）
```bash
pip install camelot-py
# 需要依赖：Ghostscript、TK（按其 README 安装）
```

### 六、阅读顺序与分栏（可选增强）

目标：尽可能按“自然阅读顺序”重建页面文本，减少“跨栏/跑偏”。

简单可行的规则：
- 取 `page.get_text("dict", sort=True)`，按 `block -> line -> span` 的 (y, x) 排序累加文本；
- 多栏检测：按页面宽度将块按 x 切分为两列，先左后右；
- 章节/标题前置：将检测到的标题/小节名拼在每个分块开头，提高检索命中。

### 七、元数据与检索过滤（建议）

- 元数据：`file_name`（keyword）、`page`（integer）
- 用途：
  - 检索过滤：如仅检索某份 PDF 或某一页
  - 答案引用：如“[文件名 - p.5]”

若要启用：
1) 在 `es_functions.py` 打开相应字段映射（启用后需重建索引）。
2) 在入库写入这些字段（`run.py` 写 ES 文档时附带）。
3) 检索时在 ES `filter` 段加入 `term/range` 条件。

### 八、推荐落地顺序

1) 电子 PDF 文本链路跑通（现有）→ 2) 图片/表格最小闭环（打印检查）→ 3) OCR 兜底（仅对空文本页）→ 4) 元数据过滤与引用 → 5) 阅读顺序与多栏优化（如确有需要）。

---

如需我直接把 OCR 兜底接入 `run.py` 的入库流程（仅空文本页触发），告诉我你的 OCR 方案（Tesseract 或 PaddleOCR），我可以代为添加最小实现。


