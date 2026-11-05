r"""Find training messages that are near-copies of check-set messages.

Why: arabic_check.csv and the training data were written by the same process, so a training message
can be a paraphrase of a check message even when the text is not identical. That inflates the check
score. This script compares meaning (embeddings, from the same cache category_model.py uses).

Run from the project root (Ollama running; embeddings are cached so it is quick):
  python ml\check_near_duplicates.py --langs en,ar --extra-dir data\synthetic
  python ml\check_near_duplicates.py --langs en,ar --extra-dir data\synthetic --threshold 0.92 --write-exclude

Writes:
  data\processed\near_duplicates.csv      every check row with its closest training row and similarity
  data\processed\exclude_train.csv        (only with --write-exclude) training texts at or above the
                                          threshold; retrain with  --exclude-file data\processed\exclude_train.csv
"""
import argparse

import numpy as np
import pandas as pd


def find_near_duplicates(Xc, X):
    """Xc: (n_check, d), X: (n_train, d), both L2-normalised. Returns best train index and cosine per check row."""
    S = Xc @ X.T
    best = S.argmax(1)
    return best, S[np.arange(len(Xc)), best]


def main():
    from category_model import load_training, cached_vectors, CHECK_PATH
    from text_utils import clean_text
    from config import PROC

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="en,ar")
    ap.add_argument("--extra-dir", default=None)
    ap.add_argument("--with-311", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--top", type=int, default=15, help="how many closest pairs to print")
    ap.add_argument("--write-exclude", action="store_true")
    a = ap.parse_args()

    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    tr = load_training(a.with_311, langs, a.extra_dir)
    ck = pd.read_csv(CHECK_PATH).dropna(subset=["text", "category"]).reset_index(drop=True)
    ck["text"] = ck["text"].map(clean_text)

    X = cached_vectors(tr["text"].tolist())
    Xc = cached_vectors(ck["text"].tolist())
    best, sim = find_near_duplicates(Xc, X)

    res = ck.copy()
    res["closest_train_text"] = tr["text"].values[best]
    res["closest_train_category"] = tr["category"].values[best]
    res["closest_train_lang"] = tr["lang"].values[best]
    res["similarity"] = sim.round(4)
    res["same_category"] = res["category"].values == res["closest_train_category"].values
    res = res.sort_values("similarity", ascending=False)
    out = PROC / "near_duplicates.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nCheck rows: {len(ck)}   training rows: {len(tr)}")
    print("\nCheck rows whose closest training row is at least this similar:")
    for t in (0.85, 0.90, 0.92, 0.95, 0.98):
        n = int((sim >= t).sum())
        print(f"  >= {t:.2f}: {n:4d}  ({n / len(ck):.1%})")
    if "source" in ck.columns:
        print(f"\nBy source, at threshold {a.threshold:.2f}:")
        for src, g in res.groupby("source"):
            n = int((g["similarity"] >= a.threshold).sum())
            print(f"  {src}: {n} of {len(g)}")
    hit = res[res["similarity"] >= a.threshold]
    if len(hit):
        print(f"\nAmong the {len(hit)} near-duplicates, the training row has the same category in "
              f"{hit['same_category'].mean():.0%} of cases.")
    print(f"\nTop {a.top} closest pairs:")
    for _, r in res.head(a.top).iterrows():
        print(f"  {r['similarity']:.3f}  check: {r['text'][:60]}\n         train: {r['closest_train_text'][:60]}")

    print(f"\nFull table saved to {out}")
    if a.write_exclude:
        ex = pd.DataFrame({"text": sorted(set(res.loc[res["similarity"] >= a.threshold, "closest_train_text"]))})
        ex_path = PROC / "exclude_train.csv"
        ex.to_csv(ex_path, index=False, encoding="utf-8-sig")
        print(f"{len(ex)} training texts written to {ex_path}")
        print("Retrain with:  --exclude-file data\\processed\\exclude_train.csv")
    elif len(hit):
        print("Add --write-exclude to list the training rows to drop, then retrain without them.")


if __name__ == "__main__":
    main()