# -*- coding: utf-8 -*-
"""
config.py —— 全局配置
项目：基于 NLP 文本测度的管理者短视与长期投资研究（毕业论文模拟稿配套代码）

说明：本文件统一管理路径、参数与变量口径，保证五步流水线可复现。
"""
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
DICT = DATA / "dictionary"
FIGURES = ROOT / "figures"

RAW_MDA = RAW / "mda"                      # 原始年报 MD&A 文本（模拟语料）
PANEL_RAW = RAW / "firm_financial_panel.csv"   # 原始公司-年度财务/治理数据

SEG_FILE = INTERIM / "segmented.jsonl"     # 分词结果
TOKEN_FILE = INTERIM / "tokens_clean.jsonl"  # 去停用词后词表
FEATURE_FILE = INTERIM / "features.npz"    # 特征矩阵
MEASURE_FILE = PROCESSED / "panel_measure.csv"   # 企业-年份面板（最终测度结果表）
MEASURE_XLSX = PROCESSED / "panel_measure.xlsx"
VALIDITY_REPORT = PROCESSED / "validity_report.txt"

# ---------- 词典 ----------
MYOPIA_DICT = DICT / "myopia_dict.txt"
STOPWORDS_CN = DICT / "stopwords_cn.txt"
STOPWORDS_FIN = DICT / "stopwords_finance.txt"

# ---------- 研究参数 ----------
YEARS = list(range(2018, 2024))     # 样本区间 2018—2023
N_FIRMS = 800                       # 模拟公司数
SEED = 20260101                     # 随机种子（保证可复现）

# 测度口径：Measure = 命中词典词总数 / 全文总词数 × 100
MEASURE_SCALE = 100.0

# 缩尾分位
WINSOR = (0.01, 0.99)

# LDA 主题数候选（用于稳健性对照）
LDA_K_RANGE = [5, 8, 10, 12, 15]
LDA_RANDOM_STATE = 42

# 词向量维度（词嵌入路线演示）
EMBED_DIM = 64

# 行业清单（模拟）
INDUSTRIES = [
    "A_农林牧渔", "C_制造业", "E_建筑业", "F_批发零售", "G_交通运输",
    "I_信息技术", "K_房地产", "L_租赁商务", "M_科学研究", "R_文化传媒",
]


def ensure_dirs():
    for p in (RAW_MDA, INTERIM, PROCESSED, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
