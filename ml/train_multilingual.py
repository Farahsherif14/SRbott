r"""Does the model understand Arabic without being trained on Arabic?

Trains a category classifier on ENGLISH data only, using multilingual bge-m3
embeddings (run through your local Ollama), then tests it on the Arabic and
Egyptian check set. Arabic is never used for training here.

Setup (PowerShell):
  ollama pull bge-m3                      # you already have it
  # save the seed file as data\processed\arabic_check.csv
  # (the name must NOT contain "pool", so make_splits.py never trains on it)
  python ml\train_multilingual.py

Embeddings are cached in data\processed\emb_cache_<model>.joblib, so the second
run takes seconds.
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from config import PROC, SEED


def post(host, path, payload, timeout=600):
    req = urllib.request.Request(f"{host}{path}", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, model, host, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        try:
            out += post(host, "/api/embed", {"model": model, "input": chunk})["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            for t in chunk:  # older Ollama versions
                out.append(post(host, "/api/embeddings", {"model": model, "prompt": t})["embedding"])
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}")
    return np.array(out, dtype=np.float32)


def get_vectors(texts, model, host, prefix=""):
    tag = model.replace(':', '_').replace('/', '_')
    if prefix:
        tag += '_' + hashlib.md5(prefix.encode()).hexdigest()[:6]
    cache_path = PROC / f"emb_cache_{tag}.joblib"
    cache = joblib.load(cache_path) if cache_path.exists() else {}
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        print(f"Embedding {len(missing):,} new texts with {model} ...")
        vecs = embed([prefix + t for t in missing], model, host)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        cache.update(dict(zip(missing, vecs)))
        joblib.dump(cache, cache_path)
    return np.stack([cache[t] for t in texts])


def load_english_train(skip_311=False):
    frames = []
    for p in sorted(PROC.glob("*pool*.csv")):
        if skip_311 and p.name == "train_pool.csv":
            print(f"skipping {p.name} (--no-311)")
            continue
        d = pd.read_csv(p)
        if not {"text", "category"} <= set(d.columns):
            continue
        if "language" in d.columns:
            d = d[d["language"] == "en"]
        if "group_id" not in d.columns:
            d["group_id"] = [f"{p.stem}_{i}" for i in range(len(d))]
        d["group_id"] = d["group_id"].astype(str)
        print(f"train source {p.name}: {len(d):,} English rows")
        frames.append(d[["text", "category", "group_id"]])
    if not frames:
        raise SystemExit(f"No English *pool*.csv data found in {PROC}")
    df = pd.concat(frames, ignore_index=True).dropna()
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--check", default=str(PROC / "arabic_check.csv"))
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="below this confidence = flagged for review")
    ap.add_argument("--prefix", default="", help="text added before every input; use \\n for a newline")
    ap.add_argument("--no-311", action="store_true", help="do not train on the templated NYC 311 rows")
    ap.add_argument("--center", action="store_true",
                    help="subtract each language's average embedding (uses Arabic text only, no labels)")
    a = ap.parse_args()
    a.prefix = a.prefix.replace("\\n", "\n")

    tr = load_english_train(a.no_311)
    print(f"\nEnglish training rows: {len(tr):,}")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    i_tr, i_va = next(gss.split(tr, groups=tr["group_id"]))
    X = get_vectors(tr["text"].tolist(), a.model, a.host, a.prefix)
    y = tr["category"].values

    ck = pd.read_csv(a.check).dropna(subset=["text", "category"])
    Xc = get_vectors(ck["text"].tolist(), a.model, a.host, a.prefix)
    if a.center:  # remove each language's average direction; labels are never used
        X = X - X.mean(0)
        Xc = Xc - Xc.mean(0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9

    clf = LogisticRegression(max_iter=3000, C=a.C, class_weight="balanced")
    clf.fit(X[i_tr], y[i_tr])
    f1_en = f1_score(y[i_va], clf.predict(X[i_va]), average="macro")
    print(f"\nEnglish validation macro-F1: {f1_en:.3f}  (inflated: templated/similar text, do not trust)")

    clf.fit(X, y)  # final model on all English data

    proba = clf.predict_proba(Xc)
    pred = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    ck["pred"], ck["conf"] = pred, conf

    print(f"\n=== Arabic/Egyptian check set ({len(ck)} rows), model never saw Arabic ===")
    print(f"category macro-F1 overall: {f1_score(ck['category'], pred, average='macro'):.3f}   (target on the real test set: 0.85)")
    if "source" in ck.columns:
        for src, g in ck.groupby("source"):
            print(f"  {src}: macro-F1 {f1_score(g['category'], g['pred'], average='macro'):.3f}  accuracy {(g['category'] == g['pred']).mean():.3f}  ({len(g)} rows)")
    print(f"flagged for review (confidence < {a.threshold}): {(conf < a.threshold).mean():.1%}")
    print("\n" + classification_report(ck["category"], pred, zero_division=0))
    labels = sorted(ck["category"].unique())
    print(pd.DataFrame(confusion_matrix(ck["category"], pred, labels=labels), index=labels, columns=labels))

    wrong = ck[ck["category"] != ck["pred"]].sort_values("conf", ascending=False)
    out = PROC / "arabic_check_errors.csv"
    wrong[["group_id", "text", "category", "pred", "conf", "source"]].to_csv(out, index=False)
    print(f"\n{len(wrong)} wrong predictions saved to {out}")


if __name__ == "__main__":
    main()