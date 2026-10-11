# -*- coding: utf-8 -*-
"""
step2_preprocessing.py —— 第二步：文本预处理（Preprocessing）

五步流水线：
    ① 文档解析   ② 文本清洗   ③ 中文分词   ④ 去停用词   ⑤ 词形归一

真实研究中 ① 常用 PyMuPDF(fitz) 抽取 PDF 正文；本演示语料为 .txt，
脚本仍保留 PDF 分支（若安装了 PyMuPDF 则自动启用），保证代码可迁移。

产出：
    data/interim/segmented.jsonl   分词结果（每行一个观测）
    data/interim/tokens_clean.jsonl 去停用词、归一后的词表
"""
import sys
import re
import json
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C

try:
    import jieba
    jieba.setLogLevel(20)
except ImportError:  # pragma: no cover
    raise SystemExit("缺少 jieba，请先 pip install -r requirements.txt")

try:  # 可选：PDF 解析
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ---------------------- ① 文档解析 ----------------------
def parse_document(fp: Path) -> str:
    if fp.suffix.lower() == ".pdf":
        if not HAS_FITZ:
            raise RuntimeError("解析 PDF 需要安装 PyMuPDF：pip install PyMuPDF")
        with fitz.open(fp) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    return fp.read_text(encoding="utf-8", errors="ignore")


# ---------------------- ② 文本清洗 ----------------------
HEADER_FOOTER = re.compile(
    r"(第\s*\d+\s*页|证券代码[:：]?\s*[\dA-Z.]{4,}|公司代码[:：]?\s*[\dA-Z.]{4,})")
NUMERIC = re.compile(r"\d{4}年|\d+\.\d+|\d+%|\d+百万元")
PUNCT = re.compile(r"[^\u4e00-\u9fa5]+")     # 仅保留中文字符，用于分词前清洗


def clean_text(text: str) -> str:
    text = HEADER_FOOTER.sub("", text)        # 去页眉页脚、代码
    text = NUMERIC.sub("", text)              # 去纯数字表达（口径统一）
    text = re.sub(r"\s+", "\n", text)         # 压缩空白
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_sentence(sent: str) -> str:
    return PUNCT.sub("", sent)                # 去标点与英文数字残留


# ---------------------- ③ 中文分词 ----------------------
def tokenize(text: str):
    """jieba 精确模式逐句分词；精确模式避免切碎专有名词。"""
    tokens = []
    for sent in text.split("\n"):
        s = normalize_sentence(sent)
        if len(s) < 2:
            continue
        seg = jieba.lcut(s, cut_all=False)     # 精确模式
        for w in seg:
            w = w.strip()
            if len(w) >= 2:                    # ⑤ 词形归一：丢弃单字与噪声
                tokens.append(w)
    return tokens


def load_stopwords() -> set:
    words = set()
    for p in (C.STOPWORDS_CN, C.STOPWORDS_FIN):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w:
                    words.add(w)
    return words


def main():
    C.ensure_dirs()
    idx = pd.read_csv(C.INTERIM / "corpus_index.csv")
    stopwords = load_stopwords()

    seg_f = open(C.INTERIM / "segmented.jsonl", "w", encoding="utf-8")
    tok_f = open(C.INTERIM / "tokens_clean.jsonl", "w", encoding="utf-8")

    n_kept_total = 0
    for _, r in idx.iterrows():
        raw = parse_document(C.ROOT / r["file_path"])          # ① 解析
        text = clean_text(raw)                                 # ② 清洗
        tokens = tokenize(text)                                # ③ 分词 + ⑤ 归一
        kept = [w for w in tokens if w not in stopwords]       # ④ 去停用词

        seg_f.write(json.dumps({"stock_code": r["stock_code"], "year": int(r["year"]),
                                "tokens": tokens}, ensure_ascii=False) + "\n")
        tok_f.write(json.dumps({"stock_code": r["stock_code"], "year": int(r["year"]),
                                "tokens": kept}, ensure_ascii=False) + "\n")
        n_kept_total += len(kept)

    seg_f.close()
    tok_f.close()

    print(f"[step2] 停用词表 {len(stopwords)} 词（通用 + 财经历）")
    print(f"[step2] 分词与清洗完成 -> {C.INTERIM/'segmented.jsonl'}")
    print(f"[step2] 有效词总数 {n_kept_total:,}，平均每篇 "
          f"{n_kept_total/len(idx):.0f} 词")


if __name__ == "__main__":
    main()
