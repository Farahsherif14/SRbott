"""Merge sources into one pool, then make grouped, stratified train/val splits.

Reads every CSV in data/processed/ named *pool*.csv (e.g. train_pool.csv,
synthetic_pool.csv). Each needs: group_id, text, category, urgency, language.
Writes data/processed/train.csv and val.csv. The hand-written test set stays
separate in data/test/test_handwritten.csv and is only checked for leakage.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import PROC, TEST, SEED, VAL_FRACTION_FOLDS, CATEGORIES, URGENCY
from text_utils import clean_text


def main():
    files = sorted(PROC.glob("*pool*.csv"))
    if not files:
        raise SystemExit("No *pool*.csv in data/processed/. Run clean_311.py first.")
    pool = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pool = pool[pool["category"].isin(CATEGORIES) & pool["urgency"].isin(URGENCY)]
    pool["text"] = pool["text"].map(clean_text)
    pool = pool.dropna(subset=["text", "group_id"]).drop_duplicates(subset=["text", "category"])
    print(f"Pool: {len(pool):,} rows from {[f.name for f in files]}")

    sgkf = StratifiedGroupKFold(n_splits=VAL_FRACTION_FOLDS, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(pool["text"], pool["category"], groups=pool["group_id"]))
    train, val = pool.iloc[tr], pool.iloc[va]

    assert not set(train["group_id"]) & set(val["group_id"]), "group leakage between train and val"

    test_path = TEST / "test_handwritten.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        if len(test):
            test_texts = set(test["text"].map(clean_text))
            overlap = test_texts & (set(train["text"]) | set(val["text"]))
            assert not overlap, f"{len(overlap)} test texts also appear in train/val: remove them from training"
            print(f"Test set: {len(test)} rows, no overlap with train/val")
    else:
        print("No hand-written test set yet (data/test/test_handwritten.csv)")

    train.to_csv(PROC / "train.csv", index=False)
    val.to_csv(PROC / "val.csv", index=False)
    print(f"train={len(train):,}  val={len(val):,}")
    print("\nTrain categories:\n", train["category"].value_counts())
    print("\nVal categories:\n", val["category"].value_counts())


if __name__ == "__main__":
    main()