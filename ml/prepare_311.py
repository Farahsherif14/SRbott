"""Turn the raw NYC 311 CSV into train.csv / val.csv.   Run from repo root:
    python ml/prepare_311.py
Reads  : data/raw/311_Service_Requests*.csv   (any file matching, read in chunks)
Writes : data/processed/train.csv, data/processed/val.csv
         data/test/test_set_TEMPLATE.csv     (columns for YOUR hand-written test set)

IMPORTANT - what this is and isn't:
* 311 has no free-text complaint, so `text` is built from Complaint Type + Descriptor.
  It is a rough English baseline. Add synthetic Arabic/Egyptian + English requests later.
* There is no "ac" category in 311. You must add AC examples yourself (synthetic).
* Urgency is a keyword heuristic. Review it against data/labeling_guide.md.
* The TEST set is NOT made here. Write it by hand and keep it out of training.
"""
import glob
import random
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_GLOB = "data/raw/311_Service_Requests*.csv"
MAX_PER_CATEGORY = 1500
SEED = 42
random.seed(SEED)

# ---- EDIT THIS MAPPING as a team --------------------------------------------------
# Complaint Type -> category. HEAT/HOT WATER is skipped on purpose (does not fit cleanly).
TYPE_TO_CATEGORY = {
    "PLUMBING": "plumbing",
    "WATER LEAK": "plumbing",
    "ELECTRIC": "electricity",
    "DOOR/WINDOW": "carpentry",
    "FLOORING/STAIRS": "carpentry",
    "PAINT/PLASTER": "general",
    "APPLIANCE": "general",
    "GENERAL": "general",
    "SAFETY": "general",
    "RODENT": "pest_control",
    "UNSANITARY CONDITION": None,  # decided by descriptor below
}

TEMPLATES = [
    "{p} in my apartment",
    "I have a problem: {p}",
    "Please fix this, {p}. It has been like this for days",
    "There is an issue in my unit: {p}",
    "Maintenance request - {p}",
    "{p}, can someone come and check?",
]

EMERGENCY = ("FLOOD", "BURST", "FIRE", "GAS", "SPARK", "COLLAPS")
HIGH = ("NO POWER", "OUTAGE", "SEWAGE", "NO WATER", "HEAVY", "CEILING")
LOW_TYPES = {"PAINT/PLASTER", "GENERAL", "APPLIANCE"}


def category_of(ctype: str, desc: str):
    if ctype == "UNSANITARY CONDITION":
        return "pest_control" if "PEST" in desc else "general"
    return TYPE_TO_CATEGORY.get(ctype)


def urgency_of(ctype: str, desc: str) -> str:
    blob = f"{ctype} {desc}"
    if any(k in blob for k in EMERGENCY):
        return "emergency"
    if any(k in blob for k in HIGH):
        return "high"
    if ctype in LOW_TYPES:
        return "low"
    return "medium"


def main():
    files = glob.glob(RAW_GLOB)
    if not files:
        sys.exit(f"No file matches {RAW_GLOB}. Did you move it to data/raw/ ?")
    path = files[0]
    print("Reading", path)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    # the export renamed these columns; accept old and new names
    type_col = next((c for c in ("Complaint Type", "Problem (formerly Complaint Type)")
                     if c in header), None)
    desc_col = next((c for c in ("Descriptor", "Problem Detail (formerly Descriptor)")
                     if c in header), None)
    if "Agency" not in header or not type_col or not desc_col:
        sys.exit(f"Could not find the needed columns. Your columns are:\n{header}")
    needed = ["Agency", type_col, desc_col]

    kept = []
    for chunk in pd.read_csv(path, usecols=needed, dtype=str, chunksize=500_000):
        chunk = chunk.fillna("").rename(
            columns={type_col: "Complaint Type", desc_col: "Descriptor"})
        chunk["Complaint Type"] = chunk["Complaint Type"].str.upper().str.strip()
        chunk["Descriptor"] = chunk["Descriptor"].str.upper().str.strip()
        mask = (chunk["Agency"] == "HPD") | (chunk["Complaint Type"] == "RODENT")
        kept.append(chunk[mask])
    df = pd.concat(kept, ignore_index=True)
    print("Housing rows kept:", len(df))
    print("Complaint types found:\n", df["Complaint Type"].value_counts().head(20).to_string())

    if df.empty:
        sys.exit("No HPD rows found. Check the Agency values in your file.")
    df["category"] = [category_of(c, d) for c, d in zip(df["Complaint Type"], df["Descriptor"])]
    df = df.dropna(subset=["category"])
    if df.empty:
        sys.exit("No complaint types matched TYPE_TO_CATEGORY. Compare the list printed "
                 "above with the mapping at the top of this script and edit it.")
    df["urgency"] = [urgency_of(c, d) for c, d in zip(df["Complaint Type"], df["Descriptor"])]

    # unique complaint+descriptor combos get several phrasings, then we sample per class
    combos = df.drop_duplicates(["Complaint Type", "Descriptor", "category", "urgency"])
    rows = []
    for _, r in combos.iterrows():
        phrase = r["Complaint Type"].lower()
        if r["Descriptor"]:
            phrase += f" ({r['Descriptor'].lower()})"
        for t in random.sample(TEMPLATES, k=len(TEMPLATES)):
            rows.append({"text": t.format(p=phrase), "language": "en",
                         "category": r["category"], "urgency": r["urgency"],
                         "source": "nyc311"})
    out = pd.DataFrame(rows).drop_duplicates("text")
    out = (out.groupby("category", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), MAX_PER_CATEGORY), random_state=SEED)))
    out.insert(0, "id", range(1, len(out) + 1))
    print("\nFinal rows per category:\n", out["category"].value_counts().to_string())
    if "ac" not in set(out["category"]):
        print("\nNOTE: no 'ac' rows - add synthetic AC requests before training.")

    train, val = train_test_split(out, test_size=0.2, random_state=SEED,
                                  stratify=out["category"])
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    pd.DataFrame(columns=["id", "text", "language", "category", "urgency", "source"]).to_csv(
        "data/test/test_set_TEMPLATE.csv", index=False)
    print(f"\nWrote train ({len(train)}) and val ({len(val)}).")


if __name__ == "__main__":
    main()