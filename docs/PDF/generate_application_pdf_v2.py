# -*- coding: utf-8 -*-
"""基于 Playwright + Chromium 的 PDF 生成器（LaTeX 级排版）。

输入：Data/docs/04-项目申请书.md
输出：Data/docs/PDF/04-项目申请书.pdf

方案：Markdown → HTML → Chromium headless → PDF
优势：
  - 完美中文支持（浏览器渲染）
  - 完美的 Markdown 解析（链接/表格/代码块/列表/引用）
  - 真正的 LaTeX 级排版（CJK + 自动换行 + 表格 + 页眉页脚）

使用方法：
    python Data/docs/PDF/generate_application_pdf_v2.py

依赖：
    pip install playwright markdown
    playwright install chromium
"""
import re
import sys
from pathlib import Path

# 路径配置：以脚本所在目录为基准
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # Data/docs/PDF -> Data/docs -> Data -> Rhino-bird

# 输入 MD：项目申请书
md_path = SCRIPT_DIR.parent / "04-项目申请书.md"
# 输出 PDF：与脚本同目录
pdf_dir = SCRIPT_DIR
pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir / "04-项目申请书.pdf"

if not md_path.exists():
    print(f"MD 文件不存在：{md_path}")
    sys.exit(1)


def md_to_html(md_text: str) -> str:
    """读取 MD 并构建完整 HTML（带样式）。"""
    # 处理 front-matter：去掉 > 引用块（用作注释）
    lines = md_text.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.startswith(">"):
            cleaned_lines.append(line.lstrip(">").strip())
        else:
            cleaned_lines.append(line)
    md_content = "\n".join(cleaned_lines)

    # 用 Python markdown 库解析（更可靠）
    import markdown
    md_extensions = [
        "tables",       # 表格
        "fenced_code",  # 代码块
        "codehilite",   # 代码高亮（可选）
        "sane_lists",   # 列表
        "toc",          # 目录（可选）
    ]
    html_body = markdown.markdown(md_content, extensions=md_extensions)

    # HTML 模板（包含完整样式）
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>犀牛鸟开源人才培养计划项目申请书</title>
<style>
  @page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @top-center {
      content: "犀牛鸟开源人才培养计划项目申请书";
      font-family: "Microsoft YaHei", "微软雅黑", "SimHei", "黑体", sans-serif;
      font-size: 9pt;
      color: #888;
      border-bottom: 1px solid #ddd;
      padding-bottom: 5px;
      width: 100%;
    }
    @bottom-center {
      content: "第 " counter(page) " 页 / 共 " counter(pages) " 页";
      font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
      font-size: 9pt;
      color: #888;
    }
    @bottom-right {
      content: "张伟林（Zviolin）· 2026";
      font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
      font-size: 8pt;
      color: #aaa;
    }
  }

  body {
    font-family: "Microsoft YaHei", "微软雅黑", "PingFang SC", "Hiragino Sans GB",
                 "Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    text-align: justify;
    margin: 0;
    padding: 0;
  }

  /* 标题层级 */
  h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #1f3864;
    margin: 0 0 0.6em 0;
    padding: 0.4em 0 0.3em;
    border-bottom: 3px solid #1f3864;
    page-break-before: always;
    page-break-after: avoid;
  }
  h1:first-of-type {
    page-break-before: avoid;
    text-align: center;
    border-bottom: none;
    padding-bottom: 0.2em;
    margin-bottom: 0.3em;
  }

  h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #2e5597;
    margin: 1.2em 0 0.4em;
    padding: 0.2em 0 0.2em 0.5em;
    border-left: 4px solid #2e5597;
    background: linear-gradient(to right, #e8f0fa, transparent);
    page-break-after: avoid;
  }

  h3 {
    font-size: 12.5pt;
    font-weight: 700;
    color: #4472c4;
    margin: 1em 0 0.3em;
    padding-left: 0.3em;
    border-left: 3px solid #4472c4;
    page-break-after: avoid;
  }

  h4 {
    font-size: 11pt;
    font-weight: 700;
    color: #333;
    margin: 0.8em 0 0.2em;
    page-break-after: avoid;
  }

  /* 段落 */
  p {
    margin: 0.3em 0 0.6em;
    text-indent: 2em;
  }
  /* 不缩进：第一个段落、表格后、列表后 */
  h1 + p, h2 + p, h3 + p, h4 + p,
  table + p, ul + p, ol + p, pre + p, blockquote + p {
    text-indent: 0;
  }
  /* 引用块不缩进 */
  blockquote p {
    text-indent: 0;
    margin: 0;
  }

  /* 表格 */
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6em 0 0.8em;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }
  th {
    background: #1f3864;
    color: #fff;
    font-weight: 700;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #1f3864;
  }
  td {
    padding: 5px 8px;
    border: 1px solid #c0c0c0;
    vertical-align: top;
  }
  tr:nth-child(even) {
    background: #f4f7fb;
  }

  /* 列表 */
  ul, ol {
    margin: 0.3em 0 0.6em;
    padding-left: 2em;
  }
  li {
    margin: 0.15em 0;
  }

  /* 引用 */
  blockquote {
    margin: 0.6em 0;
    padding: 0.5em 1em;
    background: #f4f7fb;
    border-left: 3px solid #4472c4;
    color: #444;
    font-size: 9.5pt;
  }

  /* 代码块 */
  code {
    font-family: "Consolas", "Courier New", monospace;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 9pt;
    color: #c7254e;
  }
  pre {
    background: #2b2b2b;
    color: #f8f8f2;
    padding: 0.8em 1em;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.45;
    page-break-inside: avoid;
  }
  pre code {
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }

  /* 水平线 */
  hr {
    border: none;
    border-top: 1px dashed #999;
    margin: 1.5em 0;
  }

  /* 链接 */
  a {
    color: #1f3864;
    text-decoration: none;
    border-bottom: 1px dotted #1f3864;
  }

  /* 加粗 */
  strong {
    color: #c7254e;
    font-weight: 700;
  }

  /* 斜体 */
  em {
    color: #2e5597;
    font-style: italic;
  }

  /* 状态标记 */
  .emoji {
    font-size: 12pt;
  }
</style>
</head>
<body>
""" + html_body + """
</body>
</html>"""

    return html_template


def generate_pdf():
    """用 Chromium headless 渲染 HTML 并输出 PDF。"""
    from playwright.sync_api import sync_playwright

    md_text = md_path.read_text(encoding="utf-8")
    html_content = md_to_html(md_text)

    # 临时 HTML 文件
    html_path = pdf_dir / "_04_temp.html"
    html_path.write_text(html_content, encoding="utf-8")

    with sync_playwright() as p:
        # 优先使用系统 Chrome（免下载 Chromium），失败则回退到 Playwright Chromium
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file:///{html_path.as_posix()}")
        page.wait_for_load_state("networkidle")

        # 生成 PDF（A4，display header/footer）
        page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={"top": "2cm", "right": "2cm", "bottom": "2.5cm", "left": "2cm"},
            print_background=True,
            display_header_footer=False,  # 使用 CSS @page 自定义页眉页脚
            prefer_css_page_size=True,
        )
        browser.close()

    # 删除临时 HTML
    html_path.unlink()

    print(f"PDF 生成成功：{pdf_path}")
    print(f"大小：{pdf_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    generate_pdf()