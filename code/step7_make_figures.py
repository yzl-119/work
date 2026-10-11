# -*- coding: utf-8 -*-
"""
step7_make_figures.py —— 生成论文配图

    fig1_conditional_means.png  按短视程度五等分组的长期投资均值（条件均值检验）
    fig2_coef_plot.png          主效应与分组回归系数及 95% 置信区间
    fig3_lda_k.png              LDA 主题数 K 的困惑度曲线（K 的选取依据）

产出目录：figures/
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent))
import config as CFG

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

CONTROLS = "Size + Lev + ROA + Cash + Growth + Age + Board + Dual + Big4 + TobinQ"
MAIN_C = "#1E4FA8"


def fig1(df):
    df = df.copy()
    df["grp"] = pd.qcut(df["Myopia"], 5, labels=["Q1\n(最低)", "Q2", "Q3", "Q4", "Q5\n(最高)"])
    g = df.groupby("grp", observed=True)["LongInv"].agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(range(len(g)), g["mean"], yerr=g["sem"] * 1.96, capsize=4,
           color=["#C9D8EE", "#9FBCE0", "#5B8FD9", "#2F6FC4", "#0E3F8C"], edgecolor="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index)
    ax.set_xlabel("管理者短视程度（按 Myopia 五等分组）")
    ax.set_ylabel("长期投资强度 LongInv")
    ax.set_title("图1  不同短视分组下的长期投资强度（均值 ± 95%CI）")
    for i, v in enumerate(g["mean"]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CFG.FIGURES / "fig1_conditional_means.png")
    plt.close(fig)
    print("[step7] fig1 ->", CFG.FIGURES / "fig1_conditional_means.png")


def coef_with_ci(df, formula, term):
    m = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["stock_code"]})
    return m.params[term], m.conf_int().loc[term, 0], m.conf_int().loc[term, 1]


def fig2(df):
    items = [
        ("全样本", df),
        ("国有企业", df[df.soe == 1]),
        ("非国有企业", df[df.soe == 0]),
        ("高技术行业", df[df.hitech == 1]),
        ("非高技术行业", df[df.hitech == 0]),
    ]
    names, coefs, los, his = [], [], [], []
    for name, sub in items:
        b, lo, hi = coef_with_ci(
            sub, f"LongInv ~ Myopia + {CONTROLS} + C(year) + C(stock_code)", "Myopia")
        names.append(name); coefs.append(b); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(names))[::-1]
    ax.errorbar(coefs, y, xerr=[np.array(coefs) - np.array(los),
                                np.array(his) - np.array(coefs)],
                fmt="o", color=MAIN_C, ecolor="#7F9FC9", capsize=4, ms=6)
    ax.axvline(0, color="#D9534F", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Myopia 的回归系数（95% 置信区间）")
    ax.set_title("图2  管理者短视对长期投资的影响：分样本系数")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CFG.FIGURES / "fig2_coef_plot.png")
    plt.close(fig)
    print("[step7] fig2 ->", CFG.FIGURES / "fig2_coef_plot.png")


def fig3():
    p = CFG.INTERIM / "features_lda.json"
    if not p.exists():
        print("[step7] 未找到 LDA 结果，跳过 fig3")
        return
    lda = json.loads(p.read_text(encoding="utf-8"))
    ks = [d["k"] for d in lda["k_curve"]]
    pp = [d["perplexity"] for d in lda["k_curve"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(ks, pp, "-o", color=MAIN_C, ms=5)
    ax.axvline(lda["k_best"], color="#27AE60", ls="--", lw=1,
               label=f"选定 K = {lda['k_best']}")
    ax.set_xlabel("主题数 K")
    ax.set_ylabel("困惑度 Perplexity")
    ax.set_title("图3  LDA 主题数 K 的困惑度曲线")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CFG.FIGURES / "fig3_lda_k.png")
    plt.close(fig)
    print("[step7] fig3 ->", CFG.FIGURES / "fig3_lda_k.png")


def main():
    CFG.ensure_dirs()
    df = pd.read_csv(CFG.MEASURE_FILE)
    fig1(df)
    fig2(df)
    fig3()


if __name__ == "__main__":
    main()
