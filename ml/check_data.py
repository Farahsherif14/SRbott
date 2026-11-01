"""Validate your data before training.  Run from repo root:  python ml/check_data.py

Expected files (edit PATHS if yours are named differently):
  data/processed/train.csv, data/processed/val.csv, data/test/test_set.csv
Expected columns: text, category  (optional: urgency, language, source)
"""
import sys

import pandas as pd

PATHS = {
    "train": "data/processed/train.csv",
    "val": "data/processed/val.csv",
    "test": "data/test/test_set.csv",
}
CATEGORIES = {"plumbing", "electricity", "ac", "carpentry", "pest_control", "general"}
URGENCIES = {"low", "medium", "high", "emergency"}

problems = []
frames = {}

for name, path in PATHS.items():
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        problems.append(f"[{name}] file not found: {path}")
        continue
    frames[name] = df
    print(f"\n=== {name} ({len(df)} rows) ===")

    for col in ("text", "category"):
        if col not in df.columns:
            problems.append(f"[{name}] missing column '{col}' (found: {list(df.columns)})")
    if "text" not in df.columns or "category" not in df.columns:
        continue

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        problems.append(f"[{name}] has empty text rows")
    bad = set(df["category"].dropna().unique()) - CATEGORIES
    if bad:
        problems.append(f"[{name}] unknown categories: {sorted(bad)}")
    if "urgency" in df.columns:
        bad_u = set(df["urgency"].dropna().unique()) - URGENCIES
        if bad_u:
            problems.append(f"[{name}] unknown urgency values: {sorted(bad_u)}")
    dups = df["text"].duplicated().sum()
    if dups:
        print(f"  note: {dups} duplicate texts")

    print("  category counts:\n" + df["category"].value_counts().to_string())
    if "language" in df.columns:
        print("  language counts:\n" + df["language"].value_counts().to_string())

# Leakage: test texts must not appear in train/val
if "test" in frames and "text" in frames["test"].columns:
    test_texts = set(frames["test"]["text"].astype(str).str.strip().str.lower())
    for name in ("train", "val"):
        if name in frames and "text" in frames[name].columns:
            overlap = test_texts & set(frames[name]["text"].astype(str).str.strip().str.lower())
            if overlap:
                problems.append(f"[LEAK] {len(overlap)} test texts also appear in {name}")

if "test" in frames and not 200 <= len(frames["test"]) <= 300:
    print(f"\nnote: test set has {len(frames['test'])} rows (plan says 200-300)")

print("\n" + ("PROBLEMS:\n- " + "\n- ".join(problems) if problems else "All checks passed."))
sys.exit(1 if problems else 0)