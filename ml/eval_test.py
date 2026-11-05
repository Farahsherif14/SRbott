"""Score the hand-written test set. Look at this number ONCE per finished model.

Trains on data/synthetic (same loader and model as cv_report.py) and scores every row of the
test set, which is never used for training or tuning.

Run from the repo root:   python -u ml\\eval_test.py
Options:
  --test data\\test          a folder of CSVs, or one CSV (default: data\\test)
  --model NAME              embedding model (default: DEFAULT_MODEL from cv_report.py)
  --save-model PATH         also save the two trained classifiers with joblib (for the bot)

Test CSV columns: text, language, category, urgency, source
Optional column "dialect" (en / eg / msa). If it is missing, the dialect comes from a file name
ending in _en / _eg / _msa, and otherwise from the "language" column (so Arabic shows as "ar").

Writes: data/processed/test_predictions.csv  and  data/processed/test_errors.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from cv_report import CATS, DEFAULT_MODEL, URGS, load, macro

TARGET_CATEGORY_F1 = 0.85  # success criterion from the project brief

# kind: emb = LaBSE only | emb+tfidf = both in one model | avg = average of two models
PLANS = {
    "emb": {"category": ("emb", 1.0), "urgency": ("emb", 1.0)},            # the old baseline
    "best": {"category": ("emb+tfidf", 10.0), "urgency": ("avg", 10.0)},   # winners of cv_experiments.py
}


def train_predict(kind, C, ytr, Etr, Ete, Ttr, Tte):
    def mk():
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=C)

    if kind == "emb":
        pairs = [(mk().fit(Etr, ytr), Ete)]
    elif kind == "emb+tfidf":
        pairs = [(mk().fit(hstack([csr_matrix(Etr), Ttr]).tocsr(), ytr),
                  hstack([csr_matrix(Ete), Tte]).tocsr())]
    else:  # avg
        pairs = [(mk().fit(Etr, ytr), Ete), (mk().fit(Ttr, ytr), Tte)]
    proba = sum(c.predict_proba(X) for c, X in pairs) / len(pairs)
    return pairs[0][0].classes_, proba, [c for c, _ in pairs]


def norm(s):
    return "".join(str(s).split()).lower()


def load_test(path):
    paths = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not paths:
        raise SystemExit(f"no CSV files found in {path}")
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        name = os.path.basename(p)
        m = re.search(r"_(en|eg|msa)\.csv$", name)
        if "dialect" not in df.columns:
            df["dialect"] = m.group(1) if m else df.get("language", "other")
        df["file"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    n0 = len(df)
    bad = ~(df["category"].isin(CATS) & df["urgency"].isin(URGS))
    if bad.any():
        print(f"WARNING: {bad.sum()} test row(s) have an unknown category/urgency and are skipped:")
        print(df[bad][["text", "category", "urgency", "file"]].head(10).to_string())
    df = df[~bad]
    dup = df.duplicated(subset="text")
    if dup.any():
        print(f"WARNING: {dup.sum()} duplicate test text(s) dropped")
    df = df[~dup].reset_index(drop=True)
    print(f"loaded {len(df)} test rows from {len(paths)} file(s) ({n0 - len(df)} skipped)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synthetic")
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variant", choices=list(PLANS), default="best",
                    help="best = emb+tfidf (category) and avg (urgency); emb = the old baseline")
    ap.add_argument("--save-model", default=None)
    args = ap.parse_args()

    train = load(args.train)
    test = load_test(args.test)
    if len(test) == 0:
        raise SystemExit(
            "\nThe test set has 0 usable rows, so there is nothing to score yet.\n"
            f"Look in {args.test}: the CSV file(s) there probably hold only the header line.\n"
            "Add rows (text, language, category, urgency, source, dialect), save as CSV with UTF-8, and run again.")

    # --- leakage checks: the test set must be independent of training ---
    seen = {norm(t) for t in train["text"]}
    exact = test["text"].map(norm).isin(seen)
    if exact.any():
        print(f"\nWARNING: {exact.sum()} test row(s) are identical to a training row and were removed:")
        print(test[exact]["text"].head(10).to_string())
        test = test[~exact].reset_index(drop=True)
        if len(test) == 0:
            raise SystemExit("Every test row was identical to a training row. Write new test rows.")

    print(f"\nmodel variant: {args.variant}  ->  {PLANS[args.variant]}")
    print("\nTest rows per dialect:\n", test["dialect"].value_counts().to_string())
    print("\nTest rows per category:\n", test["category"].value_counts().to_string())
    print("\nTest rows per urgency:\n", test["urgency"].value_counts().to_string())
    if len(test) < 200:
        print(f"\nNOTE: only {len(test)} test rows. The brief asks for 200-300; scores are noisy below that.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    Xtr = model.encode(train["text"].tolist(), batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)
    Xte = model.encode(test["text"].tolist(), batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    Ttr = tfidf.fit_transform(train["text"]).tocsr()   # fitted on training text only
    Tte = tfidf.transform(test["text"]).tocsr()

    sim = Xte @ Xtr.T
    best = sim.max(axis=1)
    near = best >= 0.95
    if near.any():
        print(f"\nWARNING: {near.sum()} test row(s) are almost the same as a training row (cosine >= 0.95).")
        print("They are KEPT, but they make the score optimistic. Rewrite them if there are many:")
        show = test[near].assign(similarity=np.round(best[near], 3))
        print(show[["text", "similarity"]].head(10).to_string())

    out = test.copy()
    saved = {}
    for target in ["category", "urgency"]:
        kind, C = PLANS[args.variant][target]
        classes, proba, clfs = train_predict(kind, C, train[target].values, Xtr, Xte, Ttr, Tte)
        pred = classes[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        out[f"{target}_pred"], out[f"{target}_conf"] = pred, np.round(conf, 2)
        saved[target] = (kind, clfs)

        y = test[target].values
        print("\n" + "=" * 60)
        print(f"TEST SET - {target.upper()}  macro-F1 = {macro(y, pred):.3f}   "
              f"accuracy = {(y == pred).mean():.3f}   rows = {len(y)}")
        print("=" * 60)
        if target == "category":
            gap = TARGET_CATEGORY_F1 - macro(y, pred)
            if gap <= 0:
                print(f"Target {TARGET_CATEGORY_F1:.2f}: REACHED")
            else:
                print(f"Target {TARGET_CATEGORY_F1:.2f}: not reached yet, {gap:.3f} short")
        print(classification_report(y, pred, zero_division=0))

        print(f"{target} macro-F1 by dialect:")
        for d in sorted(test["dialect"].unique()):
            idx = (test["dialect"] == d).values
            print(f"  {d:6s} rows={idx.sum():4d}  macro-F1={macro(y[idx], pred[idx]):.3f}  "
                  f"accuracy={(y[idx] == pred[idx]).mean():.3f}")

        if target == "category":
            print("\nConfusion matrix (rows = true, columns = predicted)")
            cm = confusion_matrix(y, pred, labels=CATS)
            print(pd.DataFrame(cm, index=CATS, columns=CATS).to_string())
            print("\nConfidence check:")
            for t in [0.3, 0.4, 0.5, 0.6]:
                keep = conf >= t
                if keep.sum():
                    print(f"  conf >= {t:.1f}: keeps {keep.mean() * 100:5.1f}% of rows, "
                          f"accuracy on those = {(y[keep] == pred[keep]).mean() * 100:5.1f}%")

    # --- the review rule discussed for the bot ---
    review = (out["category_conf"] < 0.5) | (out["urgency_pred"] == "emergency")
    auto = ~review
    print("\n" + "=" * 60)
    print("REVIEW RULE: send to a human if category confidence < 0.5 OR predicted urgency = emergency")
    print("=" * 60)
    print(f"  sent to review: {review.mean() * 100:.1f}% of rows")
    if auto.any():
        ok = (out.loc[auto, "category"] == out.loc[auto, "category_pred"]).mean() * 100
        print(f"  category accuracy on the auto-handled rows: {ok:.1f}%")
    emer = out["urgency"] == "emergency"
    if emer.any():
        caught = ((out["urgency_pred"] == "emergency") | (out["category_conf"] < 0.5))[emer].mean() * 100
        print(f"  real emergencies that reach a human: {caught:.1f}% of {emer.sum()}")

    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/test_predictions.csv", index=False, encoding="utf-8-sig")
    wrong = out[(out["category"] != out["category_pred"]) | (out["urgency"] != out["urgency_pred"])]
    wrong.sort_values(["category", "category_pred"]).to_csv(
        "data/processed/test_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(wrong)} row(s) with a wrong category or urgency saved to data/processed/test_errors.csv")

    if args.save_model:
        import joblib
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        joblib.dump({"embedding_model": args.model, "variant": args.variant, "tfidf": tfidf,
                     "category": saved["category"], "urgency": saved["urgency"]}, args.save_model)
        print(f"classifiers saved to {args.save_model}")


if __name__ == "__main__":
    main()