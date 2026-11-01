"""5-fold cross-validation report on data/synthetic/*.csv.

Why: the validation set is only ~80 rows, so one wrong row moves the score a lot.
Cross-validation tests on every row once, so the number is steadier.

Notes:
- Only data/synthetic is used. The NYC 311 template rows are NOT used.
- Run from the repo root:  python -u ml/cv_report.py
- Writes the wrong rows to data/processed/cv_errors.csv (open it in Excel).
- The contrast files contain the same sentence in English / Egyptian / MSA. To stop a
  translation from sitting in the test fold while its twin is in training, translations are
  linked into groups (same labels + very similar multilingual embedding) and each group is
  kept in ONE fold (StratifiedGroupKFold). Use --no-group to get the old, optimistic number.
- The hand-written test set is still the real judge.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

CATS = ["ac", "carpentry", "electricity", "general", "pest_control", "plumbing"]
URGS = ["emergency", "high", "low", "medium"]
DEFAULT_MODEL = "sentence-transformers/LaBSE"


def load(folder):
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = pd.read_csv(path, encoding="utf-8-sig")
        name = os.path.basename(path)
        m = re.search(r"_(en|eg|msa)\.csv$", name)
        df["dialect"] = m.group(1) if m else "other"
        df["file"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    n0 = len(df)
    df = df[df["category"].isin(CATS) & df["urgency"].isin(URGS)]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"loaded {len(df)} rows from {len(frames)} files (dropped {n0 - len(df)} bad/duplicate)")
    return df


def make_groups(df, X, threshold, near_dup=None):
    """Link translations of the same scenario. Two rows from different dialects are linked when
    they have the same category AND urgency and are each other's best match (cosine on
    normalized embeddings) above `threshold`. Returns an int group id per row."""
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sim = X @ X.T
    dial = np.asarray(df["dialect"].tolist(), dtype=object)
    lab = np.asarray((df["category"] + "|" + df["urgency"]).tolist(), dtype=object)
    for i in range(n):
        for d in set(dial) - {dial[i]}:
            cand = np.where((dial == d) & (lab == lab[i]))[0]
            if len(cand) == 0:
                continue
            j = cand[sim[i, cand].argmax()]
            if sim[i, j] < threshold:
                continue
            # mutual best match, so one sentence cannot chain unrelated ones together
            back = np.where((dial == dial[i]) & (lab == lab[j]))[0]
            if back[sim[j, back].argmax()] == i:
                parent[find(i)] = find(j)
    if near_dup:
        # also keep near-identical rewordings WITHIN one dialect together (they leak too,
        # especially for character n-gram models)
        same = dial[:, None] == dial[None, :]
        for i, j in np.argwhere(np.triu((sim >= near_dup) & same, 1)):
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    return pd.factorize(roots)[0]


def cv_predict(X, y, seed, groups=None):
    y = np.asarray(y)
    k = int(min(5, pd.Series(y).value_counts().min()))
    if groups is None:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    else:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")
    classes = np.unique(y)
    return classes[proba.argmax(axis=1)], proba.max(axis=1)


def macro(y_true, y_pred):
    labels = sorted(set(y_true))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/synthetic")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-group", action="store_true",
                    help="plain StratifiedKFold (translations may leak across folds)")
    ap.add_argument("--near-dup-threshold", type=float, default=None,
                    help="also group same-dialect rewordings above this cosine (try 0.92)")
    ap.add_argument("--group-threshold", type=float, default=0.80,
                    help="min cosine similarity to treat two dialect rows as one scenario")
    args = ap.parse_args()

    df = load(args.folder)
    print("\nRows per dialect:\n", df["dialect"].value_counts().to_string())
    print("\nRows per category:\n", df["category"].value_counts().to_string())

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    X = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True,
                     normalize_embeddings=True)

    groups = None
    if not args.no_group:
        groups = make_groups(df, X, args.group_threshold, args.near_dup_threshold)
        sizes = pd.Series(groups).value_counts()
        print(f"\nscenario groups: {len(sizes)} groups, {(sizes > 1).sum()} with 2+ translations "
              f"(largest = {sizes.max()}). Each group stays in one fold.")
        print("If 'largest' is above 3, raise --group-threshold (rows are being chained together).")

    for target in ["category", "urgency"]:
        y = df[target].values
        scores = []
        first = None
        for seed in range(args.repeats):
            pred, conf = cv_predict(X, y, seed, groups)
            scores.append(macro(y, pred))
            if first is None:
                first = (pred, conf)
        pred, conf = first
        print("\n" + "=" * 60)
        print(f"{target.upper()} ({'plain' if groups is None else 'grouped'} folds)  "
              f"macro-F1 over {args.repeats} CV runs: "
              f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}  (runs: "
              + ", ".join(f"{s:.3f}" for s in scores) + ")")
        print("=" * 60)
        print(classification_report(y, pred, zero_division=0))

        print(f"{target} macro-F1 by dialect (en = English, eg = Egyptian, msa = formal Arabic):")
        for d in ["en", "eg", "msa"]:
            idx = (df["dialect"] == d).values
            if idx.sum():
                print(f"  {d:4s} rows={idx.sum():4d}  macro-F1={macro(y[idx], pred[idx]):.3f}  "
                      f"accuracy={(y[idx] == pred[idx]).mean():.3f}")

        if target == "category":
            print("\nConfusion matrix (rows = true, columns = predicted)")
            cm = confusion_matrix(y, pred, labels=CATS)
            print(pd.DataFrame(cm, index=CATS, columns=CATS).to_string())

            print("\nConfidence check (for the future 'needs review' threshold):")
            for t in [0.3, 0.4, 0.5, 0.6]:
                keep = conf >= t
                if keep.sum():
                    print(f"  conf >= {t:.1f}: keeps {keep.mean() * 100:5.1f}% of rows, "
                          f"accuracy on those = {(y[keep] == pred[keep]).mean() * 100:5.1f}%")

            wrong = df.assign(pred=pred, conf=np.round(conf, 2))
            wrong = wrong[wrong["category"] != wrong["pred"]]
            os.makedirs("data/processed", exist_ok=True)
            out = "data/processed/cv_errors.csv"
            wrong[["text", "category", "pred", "conf", "urgency", "dialect", "file"]].sort_values(
                ["category", "pred"]).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n{len(wrong)} wrong rows saved to {out}")


if __name__ == "__main__":
    main()