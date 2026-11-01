"""Compare a few classifier variants under the SAME leak-free grouped CV as cv_report.py.

Run from the repo root:   python -u ml\\cv_experiments.py
Only data/synthetic is used. The hand-written test set is NOT touched here.

Variants (all on the same folds, 3 seeds each):
  emb C=...      LaBSE embeddings + logistic regression, different regularisation strengths
  tfidf          character TF-IDF only
  emb+tfidf      embeddings and character TF-IDF side by side in one model
  avg(emb,tfidf) average the probabilities of the two separate models
Why: embeddings capture meaning, character n-grams catch exact words and spelling variants
(useful for Egyptian spelling). Combining them is cheap and often helps.

Rule: a variant only counts as better if it beats the baseline by more than the seed spread
(about 0.01). Pick one, then check it ONCE on the test set with eval_test.py.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from cv_report import DEFAULT_MODEL, load, macro, make_groups


def run(df, X_list, y, groups, seeds, kind, C):
    """Return mean macro-F1 and the Egyptian-only macro-F1 over seeds."""
    all_scores, eg_scores = [], []
    y = np.asarray(y)
    classes = np.unique(y)
    k = int(min(5, pd.Series(y).value_counts().min()))
    for seed in range(seeds):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
        if kind == "avg":
            p = sum(cross_val_predict(clf, Xi, y, cv=cv, groups=groups, method="predict_proba")
                    for Xi in X_list) / len(X_list)
        else:
            X = X_list[0] if len(X_list) == 1 else hstack([csr_matrix(x) for x in X_list]).tocsr()
            p = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")
        pred = classes[p.argmax(axis=1)]
        all_scores.append(macro(y, pred))
        eg = (df["dialect"] == "eg").values
        eg_scores.append(macro(y[eg], pred[eg]))
    return float(np.mean(all_scores)), float(np.std(all_scores)), float(np.mean(eg_scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/synthetic")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--group-threshold", type=float, default=0.80)
    ap.add_argument("--near-dup-threshold", type=float, default=None,
                    help="also group same-dialect rewordings above this cosine (try 0.92)")
    args = ap.parse_args()

    df = load(args.folder)
    from sentence_transformers import SentenceTransformer

    E = SentenceTransformer(args.model).encode(
        df["text"].tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    groups = make_groups(df, E, args.group_threshold, args.near_dup_threshold)
    sizes = pd.Series(groups).value_counts()
    print(f"{len(set(groups))} scenario groups, largest = {sizes.max()}")

    # character n-grams: no labels are used to build them
    T = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                        min_df=2).fit_transform(df["text"]).tocsr()

    variants = [("emb C=0.3", [E], "one", 0.3), ("emb C=1 (current)", [E], "one", 1.0),
                ("emb C=3", [E], "one", 3.0), ("emb C=10", [E], "one", 10.0),
                ("emb C=30", [E], "one", 30.0), ("tfidf C=10", [T], "one", 10.0),
                ("emb+tfidf C=10", [E, T], "one", 10.0),
                ("avg(emb,tfidf) C=10", [E, T], "avg", 10.0)]

    for target in ["category", "urgency"]:
        y = df[target].values
        rows = []
        for name, X_list, kind, C in variants:
            m, s, eg = run(df, X_list, y, groups, args.seeds, kind, C)
            rows.append((name, m, s, eg))
            print(f"  {target:8s} {name:22s} macro-F1={m:.3f} +/- {s:.3f}   egyptian={eg:.3f}", flush=True)
        base = [r for r in rows if "current" in r[0]][0][1]
        best = max(rows, key=lambda r: r[1])
        print(f"\n{target.upper()}: baseline {base:.3f}; best = {best[0]} at {best[1]:.3f} "
              f"({best[1] - base:+.3f})")
        if best[1] - base < 0.01:
            print("  -> the gain is within the seed spread, so treat it as no real difference.\n")
        else:
            print("  -> a real gain. Use this variant, then check it once on the test set.\n")


if __name__ == "__main__":
    main()