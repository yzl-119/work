# -*- coding: utf-8 -*-
"""
step0_generate_demo_data.py —— 生成可公开的模拟语料与面板数据

【为什么需要这一步】
真实年报文本受版权与数据库授权限制，无法随论文公开。为了让五步流水线
"可复现、可运行"，本脚本按研究设计生成一份**结构同构的模拟数据集**：
    - 语料：data/raw/mda/{stock_code}_{year}.txt （模拟年报"管理层讨论与分析"）
    - 面板：data/raw/firm_financial_panel.csv （模拟财务/治理/行业数据）
生成规则与研究假设方向一致（管理者短视程度越高，文本中短期视域词占比越高；
短视程度越高，长期投资强度越低），从而使后续实证结果具有可解释性。

【重要声明】
本数据集为教学演示用**模拟数据**，不代表任何真实企业的真实披露内容，
不可用于任何实质性结论。正式研究请以 CSMAR / Wind / 巨潮资讯网 等
授权数据源为准（见 docs/01_数据来源说明.md）。
"""
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C

random.seed(C.SEED)
np.random.seed(C.SEED)

# ------------------------------------------------------------------
# 句库：短期视域句（含词典词）与长期战略句
# ------------------------------------------------------------------
SHORT_SENTENCES = [
    "公司将在短期内尽快完成产能布局调整，抓住当前市场窗口。",
    "为确保年度考核目标，管理层将抓紧推进在手订单的快速交付。",
    "本年度公司将聚焦短期业绩的稳定，努力实现季度利润的及时兑现。",
    "面对市场竞争，公司将立即采取措施，尽快回笼现金、加快回款。",
    "公司拟在近期通过变现部分非核心资产，迅速改善当期现金流。",
    "为控制风险，公司将暂停长期投入，优先保证短期目标与保底收益。",
    "管理层认为应尽早实现扭亏，着眼目前经营压力，抓紧优化成本结构。",
    "公司将采取应急与权宜之计，力求短期内见效，稳定投资者预期。",
    "本报告期内，公司将月度跟踪经营指标，及时调整以达成年度考核。",
    "公司将快速推出新品以获取短期回报，尽快兑现对股东的承诺。",
]

LONG_SENTENCES = [
    "公司坚持以自主创新为长期战略导向，持续加大研发投入力度。",
    "面向未来，公司将深度推进数字化转型，构建长期竞争壁垒。",
    "公司着力于人才梯队建设与组织能力沉淀，为长期发展奠定基础。",
    "围绕产业链延伸，公司将持续进行战略性资本性支出布局。",
    "公司将完善治理机制，强化长期价值创造能力的培育。",
    "公司注重核心技术积累，稳步推进专利布局与研发平台建设。",
    "公司将以绿色低碳为长期发展主线，推动可持续经营能力提升。",
    "公司持续优化研发组织架构，为未来技术路线储备关键能力。",
    "公司将依托产业协同，逐步构建难以复制的长期竞争优势。",
    "公司坚持长期主义，致力于为股东创造可持续的长期价值回报。",
]

TITLE_TMPL = "第三节 管理层讨论与分析\n\n"
LEAD = "报告期内，公司围绕既定战略开展经营。\n"


def gen_doc(myopia_level: float, base_len: int) -> str:
    """依据短视水平生成一篇模拟 MD&A 文本。

    myopia_level: 标准化后的短视水平（越大越短视，短期句占比越高）
    """
    # 短期句占比落在 0.05 ~ 0.55 之间，与短视水平正相关
    p_short = float(np.clip(0.20 + 0.13 * myopia_level, 0.03, 0.60))
    n_sent = max(12, int(np.random.normal(base_len, base_len * 0.15)))
    parts = [TITLE_TMPL, LEAD]
    for _ in range(n_sent):
        parts.append(random.choice(SHORT_SENTENCES if random.random() < p_short
                                   else LONG_SENTENCES))
        parts.append("\n")
    # 加入少量公司经营细节句（稀释，使词频口径更真实）
    for _ in range(max(3, n_sent // 8)):
        parts.append(
            f"公司实现营业收入{np.random.randint(5, 800)}百万元，较上年同期变动"
            f"{np.random.uniform(-25, 45):.1f}%。\n"
        )
    return "".join(parts)


def main():
    C.ensure_dirs()

    # ---------- 1. 公司与行业属性 ----------
    codes = [f"{600000 + i}.SH" if i % 3 == 0 else f"{i:06d}.SZ"
             for i in range(1, C.N_FIRMS + 1)]
    industries = np.random.choice(C.INDUSTRIES, size=C.N_FIRMS)
    soe = (np.random.rand(C.N_FIRMS) < 0.32).astype(int)          # 是否国有
    hitech = np.array([1 if ind in ("C_制造业", "I_信息技术", "M_科学研究") else 0
                       for ind in industries])                     # 是否高技术
    # 公司层面潜在短视倾向（潜在变量，仅用于生成数据，实证中由文本测度代理）
    latent = np.random.normal(0, 1, C.N_FIRMS)
    # 行业层面竞争程度（赫芬达尔指数的反向指标，越大竞争越激烈）
    # 注意：竞争程度随"行业-年度"变化（同一行业内不同年份亦有波动），
    # 以便在控制公司固定效应后，交互项仍有可识别的时变变异。
    ind_comp_base = {ind: float(np.random.uniform(0.20, 1.00)) for ind in C.INDUSTRIES}
    year_shock = {y: float(np.random.normal(0, 0.15)) for y in C.YEARS}

    rows = []
    for i, code in enumerate(codes):
        base_len = int(np.random.uniform(45, 80))
        prev_dev = 0.0          # 用于构造公司内部一阶自相关（AR(1)）
        prev_level = 0.0        # 上一期短视水平
        for t, year in enumerate(C.YEARS):
            # 公司内部随时间波动，且具有持续性（AR(1)）：
            # myopia_it = 公司潜在水平 + ρ·(上期偏离) + 扰动
            eps = np.random.normal(0, 0.45)
            dev = 0.78 * prev_dev + eps
            prev_dev = dev
            myopia_level = latent[i] + dev

            # ---------- 生成文本 ----------
            doc = gen_doc(myopia_level, base_len)
            (C.RAW_MDA / f"{code.replace('.', '_')}_{year}.txt").write_text(
                doc, encoding="utf-8")

            # ---------- 生成财务与治理数据 ----------
            size = np.random.normal(22.2, 1.3)                    # 总资产自然对数
            lev = float(np.clip(np.random.beta(4, 6), 0.02, 0.95))  # 资产负债率
            roa = np.random.normal(0.045, 0.06)                   # 总资产收益率
            cash = float(np.clip(np.random.normal(0.16, 0.09), 0.005, 0.7))
            growth = np.random.normal(0.13, 0.28)                 # 成长性
            age = 2026 - (1995 + int(np.random.uniform(0, 25)))   # 上市年限
            board = int(np.random.randint(5, 15))                 # 董事会规模
            dual = int(np.random.rand() < 0.28)                   # 两职合一
            big4 = int(np.random.rand() < 0.07)                   # 四大审计
            tobinq = float(np.clip(np.random.normal(2.0, 0.9), 0.4, 9))

            comp = float(np.clip(ind_comp_base[industries[i]] + year_shock[year]
                                 + np.random.normal(0, 0.10), 0.10, 1.30))

            # 长期投资决策具有延续性：受当期与上期短视倾向共同影响
            eff_myopia = 0.70 * myopia_level + 0.30 * prev_level

            # ---------- 被解释变量：长期投资强度 ----------
            # 长期投资 = (研发投入 + 资本性支出) / 总资产
            # 与短视负相关；竞争程度强化该负向关系（H2）；国企/非国企存在差异（H3）
            het = 1.0
            if soe[i] == 1:
                het = 0.55                    # 国企中负向关系明显更弱
            if hitech[i] == 1:
                het *= 1.25
            long_inv = (
                0.075
                - 0.022 * eff_myopia * het * (0.25 + comp)
                + 0.004 * (size - 22.2)
                - 0.030 * lev
                + 0.060 * cash
                + 0.008 * growth
                - 0.001 * age
                + 0.006 * tobinq / 2
                + np.random.normal(0, 0.040)
            )
            long_inv = float(np.clip(long_inv, 0.0005, 0.35))

            rows.append({
                "stock_code": code,
                "year": year,
                "industry": industries[i],
                "ind_comp": round(comp, 4),
                "soe": soe[i],
                "hitech": hitech[i],
                "size": round(size, 4),
                "lev": round(lev, 4),
                "roa": round(roa, 4),
                "cash": round(cash, 4),
                "growth": round(growth, 4),
                "age": age,
                "board": board,
                "dual": dual,
                "big4": big4,
                "tobinq": round(tobinq, 4),
                "long_inv": round(long_inv, 6),
                # 潜在真实值，仅作数据生成校验用，正式实证中不作为变量
                "latent_myopia": round(myopia_level, 4),
            })
            prev_level = myopia_level

    panel = pd.DataFrame(rows)
    panel.to_csv(C.PANEL_RAW, index=False, encoding="utf-8-sig")

    print(f"[step0] 已生成模拟语料 {len(panel)} 篇 -> {C.RAW_MDA}")
    print(f"[step0] 已生成面板数据 {panel.shape} -> {C.PANEL_RAW}")


if __name__ == "__main__":
    main()
