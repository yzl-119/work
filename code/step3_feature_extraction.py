# -*- coding: utf-8 -*-
"""
step3_feature_extraction.py —— 第三步：特征提取（Feature Extraction）

三条技术路线并列实现，互为对照：
    路线一  词典法（主测度）：命中词典词计数 → 最透明、可复现
    路线二  词嵌入：把词/文档映射为向量，用余弦相似度刻画语义（对照）
    路线三  主题模型 LDA：让语料自己"分话题"，用于印证词典维度（对照）

产出：
    data/interim/features.npz         词典法命中矩阵（稀疏友好的稠密小矩阵）
    data/interim/features_lda.json    LDA 主题数与困惑度、主题关键词
    data/interim/word_vectors.npz     词向量（词嵌入路线，若可用）
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

sys.path.append(str(Path(__file__).resolve().parent))
import config as C


def load_tokens(path, stopwords=None):
    """读取 tokens_clean.jsonl，返回 (keys, token_lists)。"""
    keys, seqs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            keys.append((obj["stock_code"], obj["year"]))
            toks = obj["tokens"]
            if stopwords:
                toks = [w for w in toks if w not in stopwords]
            seqs.append(toks)
    return keys, seqs


def load_dict_words() -> list:
    return [w.strip() for w in C.MYOPIA_DICT.read_text(encoding="utf-8").splitlines()
            if w.strip()]


# ---------------------- 路线一：词典法 ----------------------
def route_dictionary(keys, seqs, dict_words):
    """核心动作：拿词典去文本里"盖章"计数。

    Measure_raw = 命中的词典词总数 / 全文总词数 × 100
    （分母口径亦可用句频、字数，论文中须说明并做稳健性检验）
    """
    dict_set = set(dict_words)
    rows = []
    for (code, year), toks in zip(keys, seqs):
        n_total = len(toks)
        hits = sum(1 for w in toks if w in dict_set)
        ratio = (hits / n_total * C.MEASURE_SCALE) if n_total else np.nan
        rows.append({"stock_code": code, "year": year,
                     "dict_hits": hits, "total_tokens": n_total,
                     "myopia_raw": ratio})
    return pd.DataFrame(rows)


# ---------------------- 路线三：主题模型 LDA ----------------------
def route_lda(keys, seqs):
    docs = [" ".join(t) for t in seqs]
    vec = CountVectorizer(max_features=3000, min_df=5, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    curve = []
    best = None
    for k in C.LDA_K_RANGE:
        lda = LatentDirichletAllocation(n_components=k, max_iter=15,
                                        random_state=C.LDA_RANDOM_STATE,
                                        learning_method="batch")
        lda.fit(X)
        perp = float(lda.perplexity(X))
        curve.append({"k": k, "perplexity": round(perp, 2)})
        if best is None or perp < best[1]:
            best = (k, perp, lda)

    k_best, perp_best, lda = best
    topics = []
    for ti, comp in enumerate(lda.components_):
        top = vocab[np.argsort(comp)[::-1][:10]]
        topics.append({"topic": ti + 1, "top_words": list(top)})

    # 文档-主题占比矩阵（路径A：主题比例当变量；论文建议路径B：主题定词再词典法）
    doc_topic = lda.transform(X)
    np.savez_compressed(C.INTERIM / "lda_doc_topic.npz",
                        doc_topic=doc_topic,
                        keys=np.array([f"{c}|{y}" for c, y in keys]))

    out = {"k_curve": curve, "k_best": k_best,
           "perplexity_best": round(perp_best, 2), "topics": topics}
    (C.INTERIM / "features_lda.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ---------------------- 路线二：词嵌入 ----------------------
def route_embedding(keys, seqs):
    """优先使用 gensim Word2Vec；不可用时退化为"文档-词矩阵 + SVD"的潜在语义
    分析（LSA）静态向量，用于演示向量空间直觉。
    说明：LSA 向量是"共现结构"的可复现近似，不等价于 Word2Vec/GloVe/BERT，
    论文中若采用此路线须如实说明所用方法。"""
    docs = [" ".join(t) for t in seqs]
    vec = CountVectorizer(max_features=3000, min_df=5, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)                     # (n_docs, n_terms)
    vocab = list(vec.get_feature_names_out())

    try:
        from gensim.models import Word2Vec   # noqa
        model = Word2Vec(seqs, vector_size=C.EMBED_DIM, window=5, min_count=5,
                         workers=2, seed=C.SEED, epochs=5)
        vocab = list(model.wv.index_to_key)
        word_mat = np.vstack([model.wv[w] for w in vocab])
        method = "gensim.Word2Vec"

        def doc_vec(toks):
            vs = [model.wv[w] for w in toks if w in model.wv]
            return np.mean(vs, axis=0) if vs else np.zeros(C.EMBED_DIM)
        doc_matrix = np.vstack([doc_vec(t) for t in seqs])
    except Exception as e:  # 退化路径：LSA
        method = f"LSA(SVD) fallback [{type(e).__name__}]"
        k = min(C.EMBED_DIM, min(X.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=C.SEED)
        doc_matrix = svd.fit_transform(X)           # (n_docs, k) 文档向量
        word_mat = svd.components_.T                # (n_terms, k) 词向量

    np.savez_compressed(C.INTERIM / "word_vectors.npz",
                        vocab=np.array(vocab), vectors=word_mat)
    np.savez_compressed(C.INTERIM / "doc_vectors.npz",
                        keys=np.array([f"{c}|{y}" for c, y in keys]),
                        vectors=doc_matrix)
    return {"method": method, "vocab_size": len(vocab),
            "dim": doc_matrix.shape[1]}


def main():
    C.ensure_dirs()
    keys, seqs = load_tokens(C.INTERIM / "tokens_clean.jsonl")
    dict_words = load_dict_words()

    # 路线一
    df_dict = route_dictionary(keys, seqs, dict_words)
    df_dict.to_csv(C.INTERIM / "features_dict.csv", index=False, encoding="utf-8-sig")
    print(f"[step3] 词典法：{len(dict_words)} 词典词，"
          f"平均命中 {df_dict['dict_hits'].mean():.1f} 次，"
          f"平均测度 {df_dict['myopia_raw'].mean():.3f}")

    # TF-IDF 关键词（辅助说明词典词是否为文本高频实词）
    docs = [" ".join(t) for t in seqs]
    tfidf = TfidfVectorizer(max_features=30, token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(docs)
    top_tfidf = list(tfidf.get_feature_names_out())
    print(f"[step3] TF-IDF 高频词（前10）：{top_tfidf[:10]}")

    # 路线三
    lda = route_lda(keys, seqs)
    print(f"[step3] LDA：最佳 K={lda['k_best']}（困惑度 {lda['perplexity_best']}）")
    for t in lda["topics"][:3]:
        print(f"        主题{t['topic']}: {'/'.join(t['top_words'][:6])}")

    # 路线二
    emb = route_embedding(keys, seqs)
    print(f"[step3] 词嵌入：{emb['method']}，词表 {emb['vocab_size']}，"
          f"维度 {emb['dim']}")

    # 汇总特征矩阵
    np.savez_compressed(C.FEATURE_FILE,
                        dict_hits=df_dict["dict_hits"].to_numpy(),
                        total_tokens=df_dict["total_tokens"].to_numpy(),
                        myopia_raw=df_dict["myopia_raw"].to_numpy())
    print(f"[step3] 特征矩阵 -> {C.FEATURE_FILE}")


if __name__ == "__main__":
    main()
