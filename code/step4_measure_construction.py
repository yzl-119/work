# -*- coding: utf-8 -*-
"""
step4_measure_construction.py —— 第四步：指标构建（Measure Construction）

把"文本向量/计数"变成"能进回归的研究变量"：
    L1 原始计数 → L2 标准化比率 → L3 复合测度
本研究的核心解释变量取 L2 口径：
    Myopia_it = 命中短期视域词总数 / MD&A 总词数 × 100
并完成：样本筛选 → 连续变量 1%/99% 缩尾 → 生成企业-年份面板。

产出：data/processed/panel_measure.csv / .xlsx（测度结果表，可直接导入 Stata）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def main():
    C.ensure_dirs()

    feat = pd.read_csv(C.INTERIM / "features_dict.csv")
    fin = pd.read_csv(C.PANEL_RAW)

    df = fin.merge(feat, on=["stock_code", "year"], how="inner")
    n0 = len(df)

    # ---------- 样本筛选（论文须写明规则）----------
    df = df[~df["industry"].str.startswith("J")]        # 剔除金融保险业
    df = df[df["year"].between(2018, 2023)]             # 样本区间
    df = df.dropna(subset=["myopia_raw", "long_inv", "size", "lev", "roa",
                           "cash", "growth", "age", "board", "tobinq"])
    df = df[df["total_tokens"] >= 50]                   # 剔除文本过短（词频不稳）
    df = df[(df["lev"] >= 0) & (df["lev"] <= 1)]
    df["myopia"] = winsorize(df["myopia_raw"], *C.WINSOR)
    df = df.reset_index(drop=True)

    # ---------- 变量口径与命名（体现"构念 + 计算方式"）----------
    df = df.rename(columns={
        "size": "Size", "lev": "Lev", "roa": "ROA", "cash": "Cash",
        "growth": "Growth", "age": "Age", "board": "Board", "dual": "Dual",
        "big4": "Big4", "tobinq": "TobinQ", "long_inv": "LongInv",
        "ind_comp": "Competition",
    })
    df["Myopia"] = df["myopia"].round(4)
    df["LongInv"] = df["LongInv"].round(6)

    cols = ["stock_code", "year", "industry", "soe", "hitech",
            "Myopia", "myopia_raw", "dict_hits", "total_tokens",
            "LongInv", "Size", "Lev", "ROA", "Cash", "Growth", "Age",
            "Board", "Dual", "Big4", "TobinQ", "Competition"]
    panel = df[cols].sort_values(["stock_code", "year"]).reset_index(drop=True)

    panel.to_csv(C.MEASURE_FILE, index=False, encoding="utf-8-sig")
    try:
        panel.to_excel(C.MEASURE_XLSX, index=False)
    except Exception as e:  # openpyxl 缺失时只留 CSV
        print(f"[step4] 写 xlsx 失败（{e}），已保留 CSV")

    print(f"[step4] 样本：{n0} → {len(panel)} 个企业-年度观测"
          f"（剔除 {n0-len(panel)} 条）")
    print(f"[step4] 公司数 {panel['stock_code'].nunique()}，"
          f"年度 {panel['year'].min()}—{panel['year'].max()}")
    print(f"[step4] Myopia 均值 {panel['Myopia'].mean():.3f}，"
          f"标准差 {panel['Myopia'].std():.3f}，"
          f"区间 [{panel['Myopia'].min():.3f}, {panel['Myopia'].max():.3f}]")
    print(f"[step4] 测度结果表 -> {C.MEASURE_FILE}")


if __name__ == "__main__":
    main()
