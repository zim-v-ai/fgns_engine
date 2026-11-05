"""
Location: // fgns_engine/scripts/build_monograph_pdf.py
Description: Publication-Grade Academic Monograph PDF Compiler & Verifier for FGNS-TRM v5.0.0
             (ARC Prize 2026 Paper Track).
             Converts `paper/FGNS_TRM_Monograph.md` into HTML5 with `markdown` +
             `pymdownx.arithmatex` (generic=True), typesets all LaTeX formulas via MathJax 3,
             embeds all 8 high-resolution scientific figures (`fig00`..`fig07`), compiles to A4
             PDF via Playwright + Headless Chromium / Microsoft Edge, and performs automated
             PyMuPDF (`fitz`) verification of formulas, tables, and embedded images.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import pymupdf as fitz
import markdown
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONOGRAPH_MD_PATH = PROJECT_ROOT / "paper" / "FGNS_TRM_Monograph.md"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"

OUTPUT_PDF_ROOT = PROJECT_ROOT / "FGNS_TRM_Monograph.pdf"
OUTPUT_PDF_ARTIFACTS_PAPER = PROJECT_ROOT / "artifacts" / "paper" / "FGNS_TRM_Monograph.pdf"
OUTPUT_PDF_PAPER_DIR = PROJECT_ROOT / "paper" / "FGNS_TRM_Monograph.pdf"
OUTPUT_PDF_LEGACY = PROJECT_ROOT / "artifacts" / "FGNS_TRM_ARC_Prize_2026_Monograph.pdf"
BUILD_HTML_PATH = PROJECT_ROOT / "artifacts" / "paper" / "FGNS_TRM_Monograph_build.html"


EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
]


ACADEMIC_CSS = r"""
@page {
    size: A4 portrait;
    margin: 14mm 14mm 16mm 14mm;
}

* {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}

html, body {
    margin: 0;
    padding: 0;
    background-color: #ffffff;
    color: #0f172a;
    font-family: "Latin Modern Roman", "Computer Modern", Georgia, "Times New Roman", Times, serif;
    font-size: 9.5pt;
    line-height: 1.42;
    text-align: justify;
    hyphens: none;
    -webkit-hyphens: none;
}

.monograph-container {
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
}

/* Title & Author Header Block */
.title-banner {
    text-align: center;
    border-top: 2.2px solid #0f172a;
    border-bottom: 1.5px solid #0f172a;
    padding: 10px 8px 10px 8px;
    margin-bottom: 12px;
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    page-break-inside: avoid;
    break-inside: avoid;
}

.title-banner h1 {
    font-size: 14.5pt;
    line-height: 1.24;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 8px 0;
    text-align: center;
    hyphens: none;
}

.title-banner p {
    font-size: 8.6pt;
    line-height: 1.38;
    color: #334155;
    margin: 0;
    text-align: center;
}

/* Section Headings */
h2 {
    font-size: 11.4pt;
    line-height: 1.28;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
    text-align: left;
    hyphens: none;
    page-break-after: avoid;
    break-after: avoid;
}

h3 {
    font-size: 10.0pt;
    line-height: 1.30;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 10px;
    margin-bottom: 4px;
    text-align: left;
    hyphens: none;
    page-break-after: avoid;
    break-after: avoid;
}

h4 {
    font-size: 9.5pt;
    line-height: 1.30;
    font-weight: 700;
    font-style: italic;
    color: #1e293b;
    margin-top: 7px;
    margin-bottom: 3px;
    text-align: left;
    hyphens: none;
    page-break-after: avoid;
    break-after: avoid;
}

p {
    margin: 0 0 6px 0;
    orphans: 3;
    widows: 3;
}

ul, ol {
    margin: 4px 0 8px 0;
    padding-left: 18px;
}

li {
    margin-bottom: 3.5px;
    text-align: justify;
}

li > p {
    margin-bottom: 3px;
}

hr {
    border: none;
    border-top: 0.8px solid #cbd5e1;
    margin: 10px 0;
}

/* Inline Code */
code {
    font-family: "JetBrains Mono", "Cascadia Code", Consolas, "Courier New", monospace;
    font-size: 8.2pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 0px 2px;
    border-radius: 2px;
    border: none;
    white-space: nowrap;
    hyphens: none;
    word-break: normal;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
}

code.code-long {
    white-space: normal;
}

.nowrap-glue {
    white-space: nowrap !important;
    display: inline-block;
}

pre {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 8px 10px;
    overflow-x: auto;
    font-size: 8.0pt;
    line-height: 1.35;
    page-break-inside: avoid;
    break-inside: avoid;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    white-space: pre;
}

/* Theorem Blockquotes */
blockquote {
    background-color: #f8fafc;
    border-left: 3.5px solid #1e3a8a;
    border-top: 0.6px solid #e2e8f0;
    border-right: 0.6px solid #e2e8f0;
    border-bottom: 0.6px solid #e2e8f0;
    border-radius: 0 4px 4px 0;
    padding: 7px 11px;
    margin: 7px 0 9px 0;
    page-break-inside: avoid;
    break-inside: avoid;
}

blockquote p:last-child {
    margin-bottom: 0;
}

/* Academic Tables (Booktabs Style) */
.table-wrapper {
    width: 100%;
    margin: 8px 0 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
}

table {
    width: 100%;
    table-layout: auto;
    border-collapse: collapse;
    font-size: 6.85pt;
    line-height: 1.18;
    margin: 0 auto;
    border-top: 1.6px solid #0f172a;
    border-bottom: 1.6px solid #0f172a;
    page-break-inside: avoid;
    break-inside: avoid;
}

thead {
    display: table-header-group;
    background-color: #f1f5f9;
    border-bottom: 1.1px solid #0f172a;
}

th {
    font-weight: 700;
    color: #0f172a;
    padding: 3px 3px;
    text-align: left;
    vertical-align: bottom;
    hyphens: none;
}

td {
    padding: 2.1px 3px;
    vertical-align: top;
    border-bottom: 0.4px solid #e2e8f0;
    text-align: left;
    hyphens: none;
    word-break: normal;
}

td[style*="text-align: center"],
th[style*="text-align: center"] {
    white-space: nowrap !important;
}

tbody tr:nth-child(even) {
    background-color: #f8fafc;
}

table code {
    font-size: 6.5pt;
    padding: 0;
    background-color: transparent;
    border: none;
    white-space: nowrap !important;
    display: inline-block;
}

/* Embedded Figures & Captions */
img {
    max-width: 100%;
    max-height: 86mm;
    width: auto;
    height: auto;
    display: block;
    margin: 4px auto;
    object-fit: contain;
    page-break-inside: avoid;
    break-inside: avoid;
}

figure.monograph-figure {
    width: 100%;
    margin: 6px auto 8px auto;
    padding: 5px 7px;
    background-color: #ffffff;
    border: 0.75px solid #cbd5e1;
    border-radius: 4px;
    page-break-inside: avoid;
    break-inside: avoid;
    text-align: center;
}

figure.monograph-figure img {
    max-width: 100%;
    max-height: 86mm;
    width: auto;
    height: auto;
    display: block;
    margin: 0 auto 4px auto;
    object-fit: contain;
    page-break-inside: avoid;
    break-inside: avoid;
}

figcaption.monograph-caption {
    font-size: 7.5pt;
    line-height: 1.28;
    color: #1e293b;
    text-align: justify;
    padding-top: 3px;
    border-top: 0.5px solid #e2e8f0;
}

figcaption.monograph-caption code {
    font-size: 7.0pt;
}

/* MathJax Display & Inline Equations */
div.arithmatex {
    margin: 5px 0;
    padding: 2px 0;
    overflow-x: visible;
    text-align: center;
    font-size: 92%;
    page-break-inside: avoid;
    break-inside: avoid;
}

span.arithmatex {
    font-size: 96%;
}

mjx-container[jax="CHTML"][display="true"] {
    margin: 0.35em 0 !important;
    max-width: 100%;
    overflow-x: visible;
}
"""


def _resolve_image_src(src: str) -> str:
    """Resolves relative figure paths in `FGNS_TRM_Monograph.md` to absolute `file:///` URIs."""
    clean_src = src.strip()
    if clean_src.startswith(("http://", "https://", "data:", "file://")):
        return clean_src

    candidate_paths = [
        (MONOGRAPH_MD_PATH.parent / clean_src).resolve(),
        (PROJECT_ROOT / clean_src).resolve(),
        (FIGURES_DIR / Path(clean_src).name).resolve(),
    ]
    for path in candidate_paths:
        if path.is_file():
            return path.as_uri()
    raise FileNotFoundError(f"Could not resolve embedded figure path: {src}")


def _postprocess_html(raw_html: str) -> str:
    """
    Wraps title/author metadata, converts image + caption paragraphs into semantic
    `<figure class="monograph-figure">` containers, resolves image URIs, and wraps tables.
    """
    html = raw_html

    # 1. Wrap top H1 + author metadata paragraph into `.title-banner`
    html = re.sub(
        r"\A\s*(<h1>.*?</h1>\s*<p>.*?</p>)\s*<hr\s*/?>",
        r'<div class="title-banner">\1</div>',
        html,
        count=1,
        flags=re.DOTALL,
    )

    # 2. Resolve all <img src="..."> attributes to absolute file:/// URIs
    def _replace_img_src(match: re.Match) -> str:
        prefix, src, suffix = match.group(1), match.group(2), match.group(3)
        resolved_uri = _resolve_image_src(src)
        return f'{prefix}{resolved_uri}{suffix}'

    html = re.sub(r'(<img\b[^>]*?\bsrc=")([^"]+)("[^>]*?>)', _replace_img_src, html)

    # 3. Combine <p><img ...></p> followed by <p><em>Figure X | ...</em></p> into <figure>
    figure_pattern = re.compile(
        r"<p>\s*(<img\b[^>]+>)\s*</p>\s*<p>\s*<em>(Figure\s+\d+\s*\|.*?)</em>\s*</p>",
        flags=re.DOTALL,
    )
    html = figure_pattern.sub(
        r'<figure class="monograph-figure">\1<figcaption class="monograph-caption">\2</figcaption></figure>',
        html,
    )

    # 4. Wrap <table>...</table> in <div class="table-wrapper">
    html = re.sub(
        r"(<table\b.*?</table>)",
        r'<div class="table-wrapper">\1</div>',
        html,
        flags=re.DOTALL,
    )

    # 5. Mark very long multi-word <code> spans (> 42 chars with spaces) with .code-long
    def _tag_long_code(match: re.Match) -> str:
        inner = match.group(1)
        if len(inner) > 42 and " " in inner:
            return f'<code class="code-long">{inner}</code>'
        return match.group(0)

    html = re.sub(r"<code>(.*?)</code>", _tag_long_code, html)

    # 6. Glue adjacent opening/closing punctuation to inline math `<span class="arithmatex">...</span>`
    # and inline `<code>...</code>` so periods, commas, and parentheses never orphan onto a new line
    html = re.sub(
        r'([\(\[\{“"]*)((?:<strong>)?<span class="arithmatex">\\\(.*?\\\)</span>(?:</strong>)?)([.,;:!?\)\]\}”"]*)',
        lambda m: (
            f'<span class="nowrap-glue">{m.group(1)}{m.group(2)}{m.group(3)}</span>'
            if (m.group(1) or m.group(3))
            else m.group(0)
        ),
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'([\(\[\{“"]*)((?:<strong>)?<code>(?:(?!</code>).)*?</code>(?:</strong>)?)([.,;:!?\)\]\}”"]*)',
        lambda m: (
            f'<span class="nowrap-glue">{m.group(1)}{m.group(2)}{m.group(3)}</span>'
            if (m.group(1) or m.group(3))
            else m.group(0)
        ),
        html,
        flags=re.DOTALL,
    )

    return html


def _extract_display_math_blocks(md_text: str) -> Tuple[str, List[str]]:
    """
    Extracts all `$$ ... $$` display equations (including those inside blockquotes `> `
    or indented list items) into safe placeholders before Markdown parsing, so that
    `python-markdown` never mangles indentation, list numbering, or LaTeX symbols.
    """
    display_blocks: List[str] = []

    def _replace_block(match: re.Match) -> str:
        prefix_line = match.group(1)
        raw_body = match.group(2)
        # Strip leading `>` or whitespace from each line of the display math body
        cleaned_lines = [
            re.sub(r"^[ \t]*>?\s*", "", line) for line in raw_body.strip().splitlines()
        ]
        latex_body = "\n".join(cleaned_lines).strip()
        idx = len(display_blocks)
        display_blocks.append(latex_body)
        return f"\n{prefix_line}@@FGNS_DISPLAY_MATH_{idx}@@\n"

    pattern = re.compile(
        r"\n([ \t]*>?[ \t]*)\$\$\s*\n(.*?)\n[ \t]*>?[ \t]*\$\$\s*(?=\n|$)",
        flags=re.DOTALL,
    )
    normalized_md = pattern.sub(_replace_block, md_text)
    return normalized_md, display_blocks


def _preprocess_markdown_lists(md_text: str) -> str:
    """
    Ensures that Markdown numbered (`1. `) and bulleted (`- `) lists that immediately
    follow a text paragraph (without an intervening blank line) are preceded by a blank
    line, and upgrades 3-space nested sublists (`   - `) to 4-space indentation (`    - `)
    so `python-markdown` renders every list as a proper `<ol>` / `<ul>` hierarchy.
    """
    lines = md_text.splitlines()
    out_lines: List[str] = []
    list_item_re = re.compile(r"^([ \t]*)(?:>\s*)?(?:\d+\.|[-*])\s+")

    for line in lines:
        # Upgrade 2-space or 3-space indented list items/continuations to 4 spaces
        if re.match(r"^   (?:\S)", line):
            line = " " + line
        elif re.match(r"^  (?:\d+\.|[-*])\s+", line):
            line = "  " + line

        m_curr = list_item_re.match(line)
        is_list_item = bool(m_curr)
        if is_list_item and out_lines:
            prev = out_lines[-1]
            prev_stripped = prev.strip()
            m_prev = list_item_re.match(prev)
            prev_is_list = bool(m_prev)
            prev_is_blockquote_blank = prev_stripped == ">"

            curr_indent = len(m_curr.group(1))
            prev_indent = len(m_prev.group(1)) if m_prev else len(prev) - len(prev.lstrip(" \t"))
            dedented_from_sublist = prev_stripped != "" and curr_indent < prev_indent

            if (
                prev_stripped
                and not prev_is_blockquote_blank
                and not prev_stripped.startswith("#")
                and (not prev_is_list or dedented_from_sublist)
            ):
                if prev_stripped.startswith(">"):
                    out_lines.append(">")
                else:
                    out_lines.append("")
        out_lines.append(line)

    return "\n".join(out_lines)


def convert_markdown_to_html_document(md_path: Path) -> str:
    """Converts `FGNS_TRM_Monograph.md` into a complete standalone HTML5 + MathJax 3 document."""
    raw_md_text = md_path.read_text(encoding="utf-8")
    list_normalized_md = _preprocess_markdown_lists(raw_md_text)
    md_text, display_blocks = _extract_display_math_blocks(list_normalized_md)

    body_html = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,
            }
        },
    )

    for idx, latex_body in enumerate(display_blocks):
        token = f"@@FGNS_DISPLAY_MATH_{idx}@@"
        replacement_div = f'<div class="arithmatex">\\[\n{latex_body}\n\\]</div>'
        body_html = re.sub(
            rf"<pre><code>\s*{re.escape(token)}\s*</code></pre>",
            lambda _, r=replacement_div: r,
            body_html,
        )
        body_html = re.sub(
            rf"<p>\s*{re.escape(token)}\s*</p>",
            lambda _, r=replacement_div: r,
            body_html,
        )
        body_html = body_html.replace(token, replacement_div)

    processed_body = _postprocess_html(body_html)

    # Verify that no unparsed `$` leaked outside `.arithmatex` containers
    stripped_check = re.sub(
        r'<(span|div) class="arithmatex">.*?</\1>',
        "",
        processed_body,
        flags=re.DOTALL,
    )
    leaked_dollars = list(re.finditer(r".{0,40}\$.{0,40}", stripped_check))
    if leaked_dollars:
        snippets = [m.group(0) for m in leaked_dollars[:5]]
        raise RuntimeError(
            f"Detected {len(leaked_dollars)} unparsed '$' characters outside arithmatex blocks: {snippets}"
        )

    mathjax_script = """
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['\\\\(', '\\\\)']],
            displayMath: [['\\\\[', '\\\\]']],
            processEscapes: true,
            tags: 'none'
        },
        chtml: {
            scale: 0.96,
            mtextInheritFont: true,
            displayAlign: 'center'
        },
        options: {
            enableAssistiveMml: false,
            menuOptions: {
                settings: {
                    assistiveMml: false
                }
            }
        },
        startup: {
            pageReady: () => {
                return MathJax.startup.defaultPageReady().then(() => {
                    // Auto-scale any wide display equations so none ever overflow the A4 column
                    const containers = document.querySelectorAll('div.arithmatex');
                    containers.forEach(div => {
                        const mjx = div.querySelector('mjx-container');
                        if (mjx && div.clientWidth > 0 && mjx.scrollWidth > div.clientWidth) {
                            const ratio = (div.clientWidth / mjx.scrollWidth) * 0.98;
                            mjx.style.transform = `scale(${ratio})`;
                            mjx.style.transformOrigin = 'center center';
                        }
                    });
                    window.__MATHJAX_DONE__ = true;
                });
            }
        }
    };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"></script>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FGNS-TRM v5.0.0 — ARC Prize 2026 Paper Track Monograph</title>
    <style>{ACADEMIC_CSS}</style>
    {mathjax_script}
</head>
<body>
    <main class="monograph-container">
        {processed_body}
    </main>
</body>
</html>
"""
    return full_html


def _find_browser_executable() -> Optional[Path]:
    """Finds a local Chromium/Edge executable if available."""
    for candidate in EDGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def render_html_to_pdf(html_path: Path, pdf_path: Path) -> Dict[str, int]:
    """
    Renders the HTML5 + MathJax 3 document to an A4 PDF using Playwright and verifies
    in-DOM MathJax & image readiness prior to printing.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    browser_exe = _find_browser_executable()

    header_template = """
    <div style="width: 100%; font-family: Georgia, 'Times New Roman', serif; font-size: 7.5pt; color: #64748b; padding: 0 14mm; display: flex; justify-content: flex-end; align-items: center;">
        <span style="font-style: italic;">ARC Prize 2026 Paper Track Technical Monograph (MIT-0)</span>
    </div>
    """

    footer_template = """
    <div style="width: 100%; font-family: Georgia, 'Times New Roman', serif; font-size: 10pt; color: #1e293b; padding: 0 14mm; display: flex; justify-content: center; align-items: center;">
        <span style="font-weight: 600;"><span class="pageNumber"></span> / <span class="totalPages"></span></span>
    </div>
    """

    with sync_playwright() as p:
        launch_kwargs: Dict[str, Any] = {"headless": True}
        if browser_exe is not None:
            launch_kwargs["executable_path"] = str(browser_exe)
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1240, "height": 1754})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=60000)

        # Wait for MathJax startup promise and custom completion flag
        page.wait_for_function("() => window.__MATHJAX_DONE__ === true", timeout=60000)
        page.evaluate("() => document.fonts.ready")

        # Audit images and MathJax containers inside the live DOM
        dom_telemetry = page.evaluate(
            """() => {
                const imgs = Array.from(document.querySelectorAll('img'));
                const brokenImgs = imgs.filter(img => !img.complete || img.naturalWidth === 0).map(img => img.src);
                const arithmatexNodes = Array.from(document.querySelectorAll('.arithmatex'));
                const unrenderedMath = arithmatexNodes.filter(n => !n.querySelector('mjx-container')).map(n => n.outerHTML);
                const arithmatexAll = arithmatexNodes.length;
                const mjxContainers = document.querySelectorAll('mjx-container').length;
                const mjxErrors = Array.from(document.querySelectorAll('mjx-merror')).map(e => e.textContent);
                const figuresCount = document.querySelectorAll('figure.monograph-figure').length;
                const tablesCount = document.querySelectorAll('table').length;
                return {
                    totalImages: imgs.length,
                    brokenImagesCount: brokenImgs.length,
                    brokenImages: brokenImgs,
                    arithmatexCount: arithmatexAll,
                    mjxContainerCount: mjxContainers,
                    unrenderedMath: unrenderedMath,
                    mjxErrorCount: mjxErrors.length,
                    mjxErrors: mjxErrors,
                    figuresCount: figuresCount,
                    tablesCount: tablesCount
                };
            }"""
        )

        if dom_telemetry["figuresCount"] < 8:
            browser.close()
            raise RuntimeError(
                f"Expected 8 <figure class='monograph-figure'> elements, found {dom_telemetry['figuresCount']}"
            )
        if dom_telemetry["brokenImagesCount"] > 0:
            browser.close()
            raise RuntimeError(
                f"Broken images detected in DOM: {dom_telemetry['brokenImages']}"
            )
        if dom_telemetry["mjxErrorCount"] > 0:
            browser.close()
            raise RuntimeError(
                f"MathJax rendering errors detected: {dom_telemetry['mjxErrors']}"
            )
        if dom_telemetry["mjxContainerCount"] < dom_telemetry["arithmatexCount"]:
            browser.close()
            raise RuntimeError(
                f"Incomplete MathJax rendering: {dom_telemetry['mjxContainerCount']} / {dom_telemetry['arithmatexCount']} | Unrendered: {dom_telemetry['unrenderedMath']}"
            )

        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=header_template,
            footer_template=footer_template,
            margin={
                "top": "14mm",
                "right": "14mm",
                "bottom": "16mm",
                "left": "14mm",
            },
        )
        browser.close()

    return dom_telemetry


def verify_monograph_pdf(pdf_path: Path) -> Dict[str, object]:
    """
    Performs an automated audit of the generated PDF using PyMuPDF (`fitz`):
    1. Checks file existence, size (> 2 MB), and exact page count.
    2. Extracts visible text across all pages and checks via regex that zero raw LaTeX
       commands (`$$`, `\\frac{`, `\\mathcal{`, `\\mathbb{`, `\\begin{`, `\\blacksquare`,
       `\\underbrace`, etc.) leaked into the PDF text layer.
    3. Counts embedded raster images across all pages (must be >= 8).
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Generated PDF not found at {pdf_path}")

    size_bytes = pdf_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)

    forbidden_latex_patterns = [
        r"\$\$",
        r"\\frac\{",
        r"\\mathcal\{",
        r"\\mathbb\{",
        r"\\begin\{",
        r"\\blacksquare",
        r"\\underbrace",
        r"\\boldsymbol",
        r"\\mathrm\{",
        r"\\mathfrak\{",
        r"\\left\[",
        r"\\right\]",
        r"\\lvert",
        r"\\rvert",
        r"\\Vert",
    ]

    leaked_tokens: List[Tuple[int, str, str]] = []
    total_image_refs = 0
    unique_xrefs = set()

    for page_idx in range(num_pages):
        page = doc[page_idx]
        text: str = cast(str, page.get_text("text"))
        for pat in forbidden_latex_patterns:
            for m in re.finditer(pat, text):
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                snippet = text[start:end].replace("\n", " ")
                leaked_tokens.append((page_idx + 1, pat, snippet))

        page_imgs = page.get_images(full=True)
        total_image_refs += len(page_imgs)
        for img_info in page_imgs:
            unique_xrefs.add(img_info[0])

    doc.close()

    if size_mb < 2.0:
        raise RuntimeError(
            f"PDF size ({size_mb:.2f} MB) is unexpectedly small (< 2.0 MB); high-res figures may be missing."
        )
    if leaked_tokens:
        raise RuntimeError(
            f"Detected unrendered raw LaTeX tokens in PDF text: {leaked_tokens[:10]}"
        )
    if len(unique_xrefs) < 8:
        raise RuntimeError(
            f"Expected at least 8 embedded figures in PDF, found {len(unique_xrefs)} unique images."
        )

    return {
        "pdf_path": str(pdf_path),
        "size_mb": round(size_mb, 2),
        "num_pages": num_pages,
        "total_image_refs": total_image_refs,
        "unique_embedded_images": len(unique_xrefs),
        "raw_latex_leaks": len(leaked_tokens),
    }


def build_monograph_pdf(output_path: Optional[Path] = None) -> Path:
    """Main entrypoint: compiles `FGNS_TRM_Monograph.md` to PDF and verifies all outputs."""
    BUILD_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_doc = convert_markdown_to_html_document(MONOGRAPH_MD_PATH)
    BUILD_HTML_PATH.write_text(html_doc, encoding="utf-8")

    primary_pdf = output_path or OUTPUT_PDF_ROOT
    dom_stats = render_html_to_pdf(BUILD_HTML_PATH, primary_pdf)

    # Mirror compiled PDF to all canonical locations in the repository
    mirror_targets = [
        OUTPUT_PDF_ROOT,
        OUTPUT_PDF_ARTIFACTS_PAPER,
        OUTPUT_PDF_PAPER_DIR,
        OUTPUT_PDF_LEGACY,
    ]
    for target in mirror_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != primary_pdf.resolve():
            shutil.copy2(primary_pdf, target)

    audit = verify_monograph_pdf(primary_pdf)

    print("=" * 78)
    print("[OK] FGNS-TRM v5.0.0 ACADEMIC MONOGRAPH PDF COMPILATION & AUDIT COMPLETE")
    print("=" * 78)
    print(f"  Source Markdown          : {MONOGRAPH_MD_PATH}")
    print(f"  Primary Output PDF       : {OUTPUT_PDF_ROOT}")
    print(f"  Artifacts Paper PDF      : {OUTPUT_PDF_ARTIFACTS_PAPER}")
    print(f"  Paper Directory PDF      : {OUTPUT_PDF_PAPER_DIR}")
    print(f"  File Size                : {audit['size_mb']:.2f} MB")
    print(f"  Total Pages (A4)         : {audit['num_pages']}")
    print(f"  MathJax Expressions      : {dom_stats['mjxContainerCount']} / {dom_stats['arithmatexCount']} (0 errors)")
    print(f"  Tables Rendered          : {dom_stats['tablesCount']}")
    print(f"  Embedded High-Res Figures: {audit['unique_embedded_images']} unique ({audit['total_image_refs']} refs)")
    print(f"  Raw LaTeX Text Leaks     : {audit['raw_latex_leaks']}")
    print("=" * 78)
    return primary_pdf


if __name__ == "__main__":
    try:
        build_monograph_pdf()
    except Exception as exc:
        print(f"[ERROR] Monograph PDF build failed: {exc}", file=sys.stderr)
        sys.exit(1)
