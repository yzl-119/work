# -*- coding: utf-8 -*-
"""
step1_corpus_acquisition.py —— 第一步：语料获取（Corpus Acquisition）

职责：把"原始文本"整理成"结构化语料库"，并在论文"数据来源"小节留痕。
真实研究中本步对应：
    - 巨潮资讯网 / 公司官网 下载年报 PDF；
    - 用正则或目录规则抽取"管理层讨论与分析（MD&A）"章节；
    - 用公司代码 + 会计年度建立文件名与观测的一一对应。
本演示直接读取 data/raw/mda 下已备好的模拟语料，输出语料索引表。

产出：data/interim/corpus_index.csv
      字段：stock_code, year, file_path, char_len, n_lines, md5
"""
import sys
import hashlib
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C


def md5_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    C.ensure_dirs()
    if not C.RAW_MDA.exists():
        raise SystemExit("未找到原始语料目录，请先运行 step0_generate_demo_data.py")

    records = []
    for fp in sorted(C.RAW_MDA.glob("*.txt")):
        # 文件名形如 600000_SH_2018.txt
        stem = fp.stem
        parts = stem.split("_")
        year = int(parts[-1])
        code = "_".join(parts[:-1]).replace("_", ".")
        text = fp.read_text(encoding="utf-8", errors="ignore")
        records.append({
            "stock_code": code,
            "year": year,
            "file_path": str(fp.relative_to(C.ROOT)).replace("\\", "/"),
            "char_len": len(text),
            "n_lines": text.count("\n") + 1,
            "md5": md5_of(text),
        })

    idx = pd.DataFrame(records).sort_values(["stock_code", "year"]).reset_index(drop=True)

    # 合规与留痕检查：重复文本、缺失年份
    dup = idx["md5"].duplicated().sum()
    if dup:
        print(f"[step1] 警告：检测到 {dup} 条重复文本，请在样本筛选时处理")

    out = C.INTERIM / "corpus_index.csv"
    idx.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"[step1] 语料索引 {idx.shape[0]} 条 -> {out}")
    print(f"[step1] 覆盖公司 {idx['stock_code'].nunique()} 家，"
          f"年度区间 {idx['year'].min()}—{idx['year'].max()}")
    print(f"[step1] 平均文本长度 {idx['char_len'].mean():.0f} 字符")


if __name__ == "__main__":
    main()
