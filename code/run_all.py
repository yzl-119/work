# -*- coding: utf-8 -*-
"""
run_all.py —— 一键复现整条流水线

用法：
    python code/run_all.py

等价于依次执行：
    step0 生成模拟数据（教学演示用；接入真实数据时可跳过）
    step1 语料获取
    step2 文本预处理
    step3 特征提取（词典法 / 词嵌入 / 主题模型）
    step4 指标构建
    step5 信效度检验与描述性统计
    step6 实证分析
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

STEPS = [
    "step0_generate_demo_data",
    "step1_corpus_acquisition",
    "step2_preprocessing",
    "step3_feature_extraction",
    "step4_measure_construction",
    "step5_validity_reliability",
    "step6_empirical_analysis",
]


def main():
    t0 = time.time()
    for s in STEPS:
        print("\n" + "#" * 66)
        print(f"# 运行 {s}.py")
        print("#" * 66)
        runpy.run_path(str(HERE / f"{s}.py"), run_name="__main__")
    print(f"\n全部完成，用时 {time.time()-t0:.1f} 秒")


if __name__ == "__main__":
    main()
