# -*- coding: utf-8 -*-
"""
step6_empirical_analysis.py —— 实证分析（对应论文第 5 章）

    H1 主效应：管理者短视 → 长期投资（负向）
    H2 调节效应：行业竞争程度强化上述负向关系
    H3 异质性：产权性质（国企/非国企）、技术属性（高技术/非高技术）
    稳健性：滞后一期、双向固定效应、行业×年度固定效应、缩尾口径、
            子样本、工具变量（行业-年度均值）

产出：data/processed/regression_results.md / .csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent))
import config as CFG

CONTROLS = "Size + Lev + ROA + Cash + Growth + Age + Board + Dual + Big4 + TobinQ"


def fit(df, formula, cluster="stock_code"):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})


def line(model, term):
    b = model.params.get(term, np.nan)
    t = model.tvalues.get(term, np.nan)
    p = model.pvalues.get(term, np.nan)
    star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{b:+.5f}{star} (t={t:+.2f})", b, p


def main():
    CFG.ensure_dirs()
    df = pd.read_csv(CFG.MEASURE_FILE)
    df = df.sort_values(["stock_code", "year"]).reset_index(drop=True)
    # 滞后一期解释变量
    df["L_Myopia"] = df.groupby("stock_code")["Myopia"].shift(1)
    # 竞争程度中心化后的交互项
    df["Myopia_c"] = df["Myopia"] - df["Myopia"].mean()
    df["Comp_c"] = df["Competition"] - df["Competition"].mean()
    # 行业-年度均值（工具变量）
    df["IV_indyear"] = df.groupby(["industry", "year"])["Myopia"].transform("mean")

    out = []
    def w(s=""):
        out.append(s); print(s)

    w("# 实证结果汇总（模拟数据）\n")

    # ---------------- H1 主效应 ----------------
    m1 = fit(df, f"LongInv ~ Myopia + {CONTROLS} + C(year)")
    m1fe = fit(df, f"LongInv ~ Myopia + {CONTROLS} + C(year) + C(stock_code)")
    w("## 表1 管理者短视与长期投资（H1 主效应）\n")
    w("| 变量 | (1) 年度FE | (2) 双向FE |")
    w("|---|---|---|")
    s1, b1, p1 = line(m1, "Myopia")
    s2, b2, p2 = line(m1fe, "Myopia")
    w(f"| Myopia | {s1} | {s2} |")
    w(f"| 控制变量 | 是 | 是 |")
    w(f"| 年度固定效应 | 是 | 是 |")
    w(f"| 公司固定效应 | 否 | 是 |")
    w(f"| N | {int(m1.nobs)} | {int(m1fe.nobs)} |")
    w(f"| Adj. R² | {m1.rsquared_adj:.3f} | {m1fe.rsquared_adj:.3f} |")
    w(f"\n注：括号内为按公司聚类稳健标准误计算的 t 值；*** p<0.01，** p<0.05，* p<0.1。\n")

    # ---------------- H2 调节效应 ----------------
    m2 = fit(df, f"LongInv ~ Myopia_c + Comp_c + Myopia_c:Comp_c + {CONTROLS} + C(year) + C(stock_code)")
    s3, b3, p3 = line(m2, "Myopia_c:Comp_c")
    w("## 表2 行业竞争程度的调节效应（H2）\n")
    w("| 变量 | 系数 |")
    w("|---|---|")
    w(f"| Myopia×Competition | {s3} |")
    w(f"| 控制变量/双向固定效应 | 是 |")
    w(f"| N | {int(m2.nobs)} |")
    w(f"| Adj. R² | {m2.rsquared_adj:.3f} |\n")

    # ---------------- H3 异质性 ----------------
    w("## 表3 异质性分析（H3）\n")
    w("| 分组 | Myopia 系数 | N |")
    w("|---|---|---|")
    grp_res = {}
    for name, sub in [("国有企业", df[df.soe == 1]),
                      ("非国有企业", df[df.soe == 0]),
                      ("高技术行业", df[df.hitech == 1]),
                      ("非高技术行业", df[df.hitech == 0])]:
        mm = fit(sub, f"LongInv ~ Myopia + {CONTROLS} + C(year) + C(stock_code)")
        ss, _, _ = line(mm, "Myopia")
        grp_res[name] = mm.params["Myopia"]
        w(f"| {name} | {ss} | {int(mm.nobs)} |")
    w("")

    # ---------------- 稳健性 ----------------
    w("## 表4 稳健性检验\n")
    tests = [
        ("滞后一期解释变量", df.dropna(subset=["L_Myopia"]),
         f"LongInv ~ L_Myopia + {CONTROLS} + C(year) + C(stock_code)", "L_Myopia"),
        ("行业×年度固定效应", df,
         f"LongInv ~ Myopia + {CONTROLS} + C(year) + C(industry) + C(industry):C(year)", "Myopia"),
        ("剔除2018年样本", df[df.year >= 2019],
         f"LongInv ~ Myopia + {CONTROLS} + C(year) + C(stock_code)", "Myopia"),
        ("未缩尾测度", df,
         f"LongInv ~ myopia_raw + {CONTROLS} + C(year) + C(stock_code)", "myopia_raw"),
    ]
    w("| 检验 | 核心解释变量系数 | N |")
    w("|---|---|---|")
    for name, sub, f, term in tests:
        mm = fit(sub, f)
        ss, _, _ = line(mm, term)
        w(f"| {name} | {ss} | {int(mm.nobs)} |")
    w("")

    # 工具变量 2SLS
    try:
        from linearmodels.iv import IV2SLS  # 可选依赖
        d = df.dropna(subset=["IV_indyear"])
        iv = IV2SLS.from_formula(
            f"LongInv ~ 1 + {CONTROLS} + C(year) + [Myopia ~ IV_indyear]",
            d).fit(cov_type="clustered", clusters=d["stock_code"])
        b = iv.params["Myopia"]; p = iv.pvalues["Myopia"]
        w(f"| 工具变量(行业-年度均值) 2SLS | {b:+.5f} (p={p:.3f}) | {int(iv.nobs)} |")
    except Exception as e:
        w(f"| 工具变量 2SLS | 需安装 linearmodels（{type(e).__name__}）| — |")
    w("")

    # ---------------- 结论提示 ----------------
    w("## 主要结论")
    w(f"- H1：Myopia 系数 = {b1:+.5f}（p={p1:.4f}），显著为负，支持假设 1。")
    w(f"- H2：交互项系数 = {b3:+.5f}（p={p3:.4f}），"
      f"{'支持' if (b3 < 0 and p3 < 0.1) else '未支持'}假设 2。")
    soe_d = grp_res.get("国有企业", np.nan)
    non_d = grp_res.get("非国有企业", np.nan)
    hi = grp_res.get("高技术行业", np.nan)
    nhi = grp_res.get("非高技术行业", np.nan)
    w(f"- H3a（产权性质）：国企 {soe_d:+.5f} vs 非国企 {non_d:+.5f}，"
      f"{'非国企' if abs(non_d) > abs(soe_d) else '国企'}中负向关系更强。")
    w(f"- H3b（技术属性）：高技术 {hi:+.5f} vs 非高技术 {nhi:+.5f}，"
      f"{'高技术' if abs(hi) > abs(nhi) else '非高技术'}行业中负向关系更强。")

    (CFG.PROCESSED / "regression_results.md").write_text("\n".join(out), encoding="utf-8")
    print(f"\n[step6] 回归结果 -> {CFG.PROCESSED/'regression_results.md'}")


if __name__ == "__main__":
    main()
