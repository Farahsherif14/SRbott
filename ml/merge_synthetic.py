"""Combine NYC 311 rows + all synthetic CSVs in data/synthetic/ into train.csv / val.csv.
    python ml/merge_synthetic.py
Safe to re-run. The hand-written TEST set is never touched.
"""
import glob
import io

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORIES = {"plumbing", "electricity", "ac", "carpentry", "pest_control", "general"}
URGENCIES = {"low", "medium", "high", "emergency"}
SEED = 42


def read_pasted_csv(path):
    # tolerate ``` fences that sometimes get copied along with the CSV
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln for ln in f if not ln.strip().startswith("```")]
    return pd.read_csv(io.StringIO("".join(lines)))


# 311 rows already prepared by prepare_311.py
parts = []
for p in ("data/processed/train.csv", "data/processed/val.csv"):
    d = pd.read_csv(p)
    parts.append(d[d["source"] == "nyc311"])
nyc = pd.concat(parts, ignore_index=True)

# every synthetic file you saved
frames = []
for path in sorted(glob.glob("data/synthetic/*.csv")):
    try:
        d = read_pasted_csv(path)
        d = d[["text", "language", "category", "urgency"]].copy()
        d["source"] = "synthetic"
        frames.append(d)
        print(f"read {path}: {len(d)} rows")
    except Exception as e:
        print(f"SKIPPED {path}: {e}")
if not frames:
    raise SystemExit("No CSV files found in data/synthetic/")
syn = pd.concat(frames, ignore_index=True)

# clean
syn["text"] = syn["text"].astype(str).str.strip()
syn["category"] = syn["category"].astype(str).str.strip().str.lower()
syn["urgency"] = syn["urgency"].astype(str).str.strip().str.lower()
before = len(syn)
syn = syn[syn["category"].isin(CATEGORIES) & syn["urgency"].isin(URGENCIES)]
syn = syn[syn["text"].str.len() > 2].drop_duplicates("text")
print(f"synthetic rows kept: {len(syn)} of {before} (bad labels/duplicates dropped)")

df = pd.concat([nyc, syn], ignore_index=True).drop_duplicates("text")
df = df.drop(columns=["id"], errors="ignore")
df.insert(0, "id", range(1, len(df) + 1))

print("\nRows per category:\n", df["category"].value_counts().to_string())
print("\nRows per language:\n", df["language"].value_counts().to_string())

train, val = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])
train.to_csv("data/processed/train.csv", index=False, encoding="utf-8-sig")
val.to_csv("data/processed/val.csv", index=False, encoding="utf-8-sig")
print(f"\nWrote train ({len(train)}) and val ({len(val)}).")