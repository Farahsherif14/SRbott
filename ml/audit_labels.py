r"""Flag likely wrong labels with grouped cross-validation.

How it works: every row gets a prediction from a model that never saw its
group (translations share a group_id, so they stay together). If that model
confidently disagrees with the row's label, the row is flagged.

  python ml\audit_labels.py                  # report only, writes the flagged CSVs
  python ml\audit_labels.py --apply          # also remove flagged CATEGORY rows
  python ml\audit_labels.py --min-conf 0.7   # stricter: fewer, more certain flags

Reads every data\processed\*pool*.csv. Writes:
  data\processed\audit_category_flagged.csv   (review this, then --apply)
  data\processed\audit_urgency_flagged.csv    (report only, never removed)
--apply backs each changed file up as <name>.csv.bak (or .bak2, .bak3 ... if a
backup already exists) before removing rows.
"""
import argparse
import shutil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from config import PROC, SEED


def load_pools():
    frames = []
    for p in sorted(PROC.glob("*pool*.csv")):
        d = pd.read_csv(p)
        if not {"text", "category", "urgency"} <= set(d.columns):
            print(f"skipping {p.name} (missing text/category/urgency columns)")
            continue
        d["_file"] = p.name
        d["_row"] = d.index
        frames.append(d)
        print(f"loaded {p.name}: {len(d):,} rows")
    if not frames:
        raise SystemExit(f"No *pool*.csv files found in {PROC}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "category", "urgency"]).reset_index(drop=True)
    if "group_id" not in df.columns:
        df["group_id"] = np.nan
    missing = df["group_id"].isna()
    df.loc[missing, "group_id"] = [f"auto_{i}" for i in df.index[missing]]
    for col in ("language", "source"):
        if col not in df.columns:
            df[col] = ""
    return df


def oof_predict(df, target, n_splits):
    """Out-of-fold predictions: (predicted label, confidence, probability of the given label)."""
    X = df["text"].astype(str).values
    y = df[target].values
    classes = np.array(sorted(df[target].unique()))
    probs = np.zeros((len(df), len(classes)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y, groups=df["group_id"].values):
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
            LogisticRegression(max_iter=2000, C=5, class_weight="balanced"),
        )
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])
        for j, c in enumerate(model.classes_):
            probs[te, int(np.where(classes == c)[0][0])] = p[:, j]
    pred = classes[probs.argmax(1)]
    conf = probs.max(1)
    p_label = probs[np.arange(len(df)), np.searchsorted(classes, y)]
    return pred, conf, p_label


def audit(df, target, n_splits, min_conf):
    pred, conf, p_label = oof_predict(df, target, n_splits)
    out = df.copy()
    out["predicted"] = pred
    out["confidence"] = conf.round(3)
    out["p_given_label"] = p_label.round(3)
    acc = (pred == df[target].values).mean()
    flagged = out[(out["predicted"] != out[target]) & (out["confidence"] >= min_conf)]
    flagged = flagged.sort_values("confidence", ascending=False)
    print(f"\n[{target}] out-of-fold accuracy: {acc:.3f}   flagged: {len(flagged)} of {len(df)} (min-conf {min_conf})")
    if len(flagged):
        print(f"  flagged by file:\n{flagged['_file'].value_counts().to_string()}")
        print("  most common label -> predicted pairs:")
        pairs = flagged.groupby([target, "predicted"]).size().sort_values(ascending=False).head(8)
        print(pairs.to_string())
    cols = ["_file", "_row", "group_id", "text", target, "predicted", "confidence", "p_given_label", "language", "source"]
    return flagged[cols]


def apply_removals(flagged):
    for fname, g in flagged.groupby("_file"):
        path = PROC / fname
        bak = path.with_name(path.name + ".bak")
        k = 2
        while bak.exists():
            bak = path.with_name(f"{path.name}.bak{k}")
            k += 1
        shutil.copy(path, bak)
        d = pd.read_csv(path)
        before = len(d)
        d = d.drop(index=g["_row"].astype(int).tolist(), errors="ignore")
        d.to_csv(path, index=False)
        print(f"  {fname}: removed {before - len(d)} rows ({len(d)} left), backup -> {bak.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="remove flagged CATEGORY rows (urgency is report-only)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="flag only if the model is at least this sure")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    df = load_pools()
    print(f"\nTotal: {len(df):,} rows, {df['group_id'].nunique():,} groups")

    cat_flagged = audit(df, "category", a.folds, a.min_conf)
    cat_path = PROC / "audit_category_flagged.csv"
    cat_flagged.to_csv(cat_path, index=False)
    print(f"  wrote {cat_path}")

    urg_flagged = audit(df, "urgency", a.folds, a.min_conf)
    urg_path = PROC / "audit_urgency_flagged.csv"
    urg_flagged.to_csv(urg_path, index=False)
    print(f"  wrote {urg_path} (report only)")

    if a.apply:
        print("\nApplying: removing flagged category rows")
        apply_removals(cat_flagged)
    else:
        print("\nNothing changed. Review the category file, then run again with --apply.")


if __name__ == "__main__":
    main()