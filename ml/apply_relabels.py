"""Apply label decisions from labeling_guide v2 to the existing synthetic CSVs.

Run from the repo root:  python -u ml/apply_relabels.py
Safe to run twice (the second run changes nothing).
Spaces are ignored when matching, so pasted-text spacing problems do not matter.
"""
import csv
import glob
import os

import pandas as pd

# (text, new category, new urgency, reason)
RULES = [
   ("حنفية المطبخ بتنقط طول الليل وانا سامع صوتها", "plumbing", "low",
 "tie-breaker: slow tap drip is low"),
]


def norm(s):
    return "".join(str(s).split())


def main():
    wanted = {norm(t): (c, u, why) for t, c, u, why in RULES}
    total = 0
    for path in sorted(glob.glob("data/synthetic/*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        changed = False
        for i, text in df["text"].items():
            hit = wanted.get(norm(text))
            if hit and (df.at[i, "category"], df.at[i, "urgency"]) != hit[:2]:
                print(f"{os.path.basename(path)}: {df.at[i, 'category']}/{df.at[i, 'urgency']} "
                      f"-> {hit[0]}/{hit[1]}   ({hit[2]})")
                df.at[i, "category"], df.at[i, "urgency"] = hit[0], hit[1]
                changed = True
                total += 1
        if changed:
            df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL,
                      lineterminator="\r\n")
    print(f"done, {total} row(s) changed")


if __name__ == "__main__":
    main()
