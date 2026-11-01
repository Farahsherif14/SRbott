"""Turn plain-text test notes into the CSV files that eval_test.py reads.

Write your test messages in data/test/raw_en.txt, raw_eg.txt and raw_msa.txt, one per line:

    category | urgency | message text

Lines that are empty or start with # are ignored. The text may contain commas and quotes.
Categories: ac, carpentry, electricity, general, pest_control, plumbing
Urgencies : emergency, high, medium, low

Run from the repo root:   python -u ml\\build_testset.py
It (re)writes data/test/testset_en.csv, testset_eg.csv and testset_msa.csv. The .txt files are the
source of truth, so just edit them and run this again.
"""
import csv
import io
import os
import sys

CATS = {"ac", "carpentry", "electricity", "general", "pest_control", "plumbing"}
URGS = {"emergency", "high", "medium", "low"}
LANG = {"en": "en", "eg": "ar", "msa": "ar"}
FOLDER = "data/test"


def clean(label):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def main():
    problems = 0
    grand = 0
    for d, lang in LANG.items():
        src = os.path.join(FOLDER, f"raw_{d}.txt")
        if not os.path.exists(src):
            print(f"{src}: not found, skipped")
            continue
        rows, seen = [], set()
        with io.open(src, encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|", 2)]
                if len(parts) != 3 or not parts[2]:
                    print(f"{src} line {n}: need 'category | urgency | text'  ->  {line[:60]}")
                    problems += 1
                    continue
                cat, urg, text = clean(parts[0]), clean(parts[1]), parts[2]
                if cat not in CATS or urg not in URGS:
                    print(f"{src} line {n}: unknown label '{parts[0]}' / '{parts[1]}'")
                    problems += 1
                    continue
                key = "".join(text.split()).lower()
                if key in seen:
                    print(f"{src} line {n}: duplicate text, skipped  ->  {text[:60]}")
                    problems += 1
                    continue
                seen.add(key)
                rows.append([text, lang, cat, urg, "handwritten"])
        out = os.path.join(FOLDER, f"testset_{d}.csv")
        with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
            w.writerow(["text", "language", "category", "urgency", "source"])
            w.writerows(rows)
        grand += len(rows)
        by_cat = {c: sum(r[2] == c for r in rows) for c in sorted(CATS)}
        print(f"{out}: {len(rows)} rows   " + "  ".join(f"{c}={n}" for c, n in by_cat.items()))
    print(f"\ntotal test rows: {grand}   problems: {problems}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
