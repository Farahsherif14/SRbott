"""Clean NYC 311 Service Requests into the classifier training pool.

Usage:
  python src/clean_311.py --inspect                       # show columns + complaint types
  python src/clean_311.py --input data/raw/311.csv        # build data/processed/train_pool.csv

311 has NO free-text complaint, only Complaint Type + Descriptor, so the text
built here is templated. It is a starting pool; add LLM-generated natural and
Arabic requests (same group_id as their source row) before training for real.
"""
import argparse
import random
import pandas as pd

from config import RAW, PROC, SEED, MAX_PER_COMBO, CATEGORIES
from mapping import to_category, to_urgency
from text_utils import clean_text, detect_lang

SCOPE = {"ENTIRE BUILDING": "in the entire building", "APARTMENT ONLY": "in my apartment"}
TEMPLATES = [
    "I have a problem with {c} {where}: {d}.",
    "There is an issue with {c} {where}, {d}.",
    "Please help, problem with {c} {where} ({d}).",
    "Issue with {c} {where}. {d}. Can someone come and fix it?",
    "Hello, I want to report a problem with {c} {where}: {d}.",
]
TEMPLATES_NO_DESC = [
    "I have a problem with {c} {where}.",
    "Please send someone, there is an issue with {c} {where}.",
    "Hello, I want to report a problem with {c} {where}.",
]


def load(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows, low_memory=False, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # NYC renamed these columns in the current 311 export; accept old and new names
    df = df.rename(columns={
        "problem_(formerly_complaint_type)": "complaint_type",
        "problem_detail_(formerly_descriptor)": "descriptor",
    })
    if "complaint_type" not in df.columns:
        raise SystemExit(f"Missing column 'complaint_type'. Found: {list(df.columns)}")
    for col in ("descriptor", "additional_details", "unique_key", "agency"):
        if col not in df.columns:
            df[col] = ""
    df[["descriptor", "additional_details", "unique_key"]] = df[["descriptor", "additional_details", "unique_key"]].fillna("")
    return df


def build_text(ctype, desc, details, rng):
    c = ctype.strip().lower().replace("/", " / ")
    skip = set(SCOPE) | {"", "N/A", "NAN", "OTHER", "UNSPECIFIED"}
    parts = [p.strip() for p in (desc, details) if p and p.strip().upper() not in skip]
    where = next((SCOPE[p.strip().upper()] for p in (desc, details) if p and p.strip().upper() in SCOPE), "in my unit")
    if not parts:
        return rng.choice(TEMPLATES_NO_DESC).format(c=c, where=where)
    d = ", ".join(dict.fromkeys(p.lower() for p in parts))
    return rng.choice(TEMPLATES).format(c=c, where=where, d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=str(PROC / "train_pool.csv"))
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    path = a.input or next(iter(sorted(RAW.glob("*.csv"))), None)
    if path is None:
        raise SystemExit("Put the NYC 311 CSV in data/raw/ or pass --input")
    df = load(path, a.nrows)
    print(f"Loaded {len(df):,} rows from {path}")

    if a.inspect:
        print("\nColumns:", list(df.columns))
        print("\nTop complaint types:\n", df["complaint_type"].value_counts().head(40))
        return

    rng = random.Random(SEED)
    n0 = len(df)
    df = df.dropna(subset=["complaint_type"]).copy()
    df["category"] = [to_category(c, d, x) for c, d, x in zip(df["complaint_type"], df["descriptor"], df["additional_details"])]
    unmapped = df[df["category"].isna()]["complaint_type"].value_counts().head(15)
    df = df[df["category"].notna()].copy()
    print(f"Kept {len(df):,}/{n0:,} rows with a mapped category. Top dropped types:\n{unmapped}\n")

    df["urgency"] = [to_urgency(c, d, x) for c, d, x in zip(df["complaint_type"], df["descriptor"], df["additional_details"])]
    df["combo"] = df["complaint_type"].str.upper() + "|" + df["descriptor"].str.upper() + "|" + df["additional_details"].str.upper()

    # cap rows per combo so common complaints do not dominate
    df = pd.concat([g.sample(min(len(g), MAX_PER_COMBO), random_state=SEED)
                    for _, g in df.groupby("combo")])

    df["text"] = [clean_text(build_text(c, d, x, rng)) for c, d, x in zip(df["complaint_type"], df["descriptor"], df["additional_details"])]
    df["language"] = df["text"].map(detect_lang)

    before = len(df)
    df = df[df["text"].str.len() >= 10]
    df = df.drop_duplicates(subset=["text", "category", "urgency"])  # exact duplicates only
    print(f"Dropped {before - len(df):,} too-short or duplicate rows")

    # group_id: all variants (paraphrases, translations) of one source request share it
    key = df["unique_key"].where(df["unique_key"] != "", pd.Series(df.index.astype(str), index=df.index))
    df["group_id"] = "nyc311_" + key
    df["source"] = "nyc311"
    out = df[["group_id", "text", "category", "urgency", "language", "source", "combo"]].reset_index(drop=True)
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)

    print(f"\nWrote {len(out):,} rows -> {a.output}")
    counts = out["category"].value_counts()
    print("\nCategory counts:\n", counts)
    print("\nUrgency counts:\n", out["urgency"].value_counts())
    for cat in CATEGORIES:
        n = int(counts.get(cat, 0))
        if n == 0:
            print(f"WARNING: no rows for '{cat}' -> fill with synthetic data")
        elif n < 500:
            print(f"WARNING: only {n} rows for '{cat}' -> add synthetic data")


if __name__ == "__main__":
    main()