r"""Final category classifier: English-only training, multilingual embeddings.

It also predicts urgency (low / medium / high) from the same embeddings. Urgency labels in the
training data are noisy and follow the generator's rules, not the 4-level labeling guide, so treat
urgency as a rough, second-priority output.

Locked-in setup (found with train_multilingual.py):
  qwen3-embedding:4b via Ollama + instruction prefix + per-language mean centering
  + no NYC 311 rows + logistic regression (C=10).

Commands (PowerShell, from the project root):
  python ml\category_model.py train                 # train, tune review threshold, save model
  python ml\category_model.py train --mlflow        # same, and log to MLflow
  python ml\category_model.py predict "my sink is leaking" "التكييف مش بيبرد"
  python ml\category_model.py test                  # hand-written test set: run ONLY at gates

Needs: ollama running with qwen3-embedding:4b pulled, data\processed\arabic_check.csv.
The Arabic check file is used for three things only, never as labelled training data:
  1. its TEXT gives the Arabic mean embedding (centering, no labels),
  2. its labels measure the score,
  3. its labels tune the review threshold (provisional: it is a dev set, not the test set).

Use from other code:
  from category_model import CategoryClassifier
  clf = CategoryClassifier.load()
  clf.predict("الحوض مسدود")  # {'category', 'confidence', 'needs_review', 'urgency', 'urgency_confidence', 'language'}
"""
import argparse
import datetime
import hashlib
import json
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import PROC, SEED
from text_utils import clean_text, detect_lang

ROOT = PROC.parents[1]
MODEL_PATH = ROOT / "models" / "category_clf.joblib"
TEST_PATH = ROOT / "data" / "test" / "test_handwritten.csv"
CHECK_PATH = PROC / "arabic_check.csv"

HOST = "http://localhost:11434"
EMB_MODEL = "qwen3-embedding:4b"
PREFIX = "Instruct: Classify a resident maintenance request into a service category\nQuery: "
C = 10.0
URGENCY_LEVELS = ["low", "medium", "high"]


# ---------- embeddings ----------
def _post(path, payload, timeout=600):
    req = urllib.request.Request(f"{HOST}{path}", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, model=None, batch=32):
    """Raw (unnormalised) embeddings from Ollama."""
    model = model or EMB_MODEL
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        try:
            out += _post("/api/embed", {"model": model, "input": chunk})["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            for t in chunk:  # older Ollama versions
                out.append(_post("/api/embeddings", {"model": model, "prompt": t})["embedding"])
        if len(texts) > batch:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}")
    return np.array(out, dtype=np.float32)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def cached_vectors(texts):
    """Normalised embeddings with a disk cache (same cache files train_multilingual.py uses)."""
    tag = EMB_MODEL.replace(":", "_").replace("/", "_") + "_" + hashlib.md5(PREFIX.encode()).hexdigest()[:6]
    path = PROC / f"emb_cache_{tag}.joblib"
    cache = joblib.load(path) if path.exists() else {}
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        print(f"Embedding {len(missing):,} new texts with {EMB_MODEL} ...")
        vecs = _unit(embed([PREFIX + t for t in missing]))
        cache.update(dict(zip(missing, vecs)))
        joblib.dump(cache, path)
    return np.stack([cache[t] for t in texts])


# ---------- training ----------
def load_english(with_311):
    frames = []
    for p in sorted(PROC.glob("*pool*.csv")):
        if p.name == "train_pool.csv" and not with_311:
            print(f"skipping {p.name} (templated 311 text; use --with-311 to include)")
            continue
        d = pd.read_csv(p)
        if not {"text", "category"} <= set(d.columns):
            continue
        if "language" in d.columns:
            d = d[d["language"] == "en"]
        print(f"train source {p.name}: {len(d):,} English rows")
        if "urgency" not in d.columns:
            d["urgency"] = np.nan
        frames.append(d[["text", "category", "urgency"]])
    if not frames:
        raise SystemExit(f"No English *pool*.csv data found in {PROC}")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "category"])
    df["text"] = df["text"].map(clean_text)
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def threshold_table(conf, correct):
    rows = []
    for t in np.arange(0.30, 0.951, 0.05):
        keep = conf >= t
        rows.append({"threshold": round(float(t), 2),
                     "flagged": round(float(1 - keep.mean()), 3),
                     "accuracy_of_accepted": round(float(correct[keep].mean()), 3) if keep.any() else float("nan")})
    return pd.DataFrame(rows)


def pick_threshold(table, target_acc, min_coverage=0.5):
    ok = table[(table["accuracy_of_accepted"] >= target_acc) & (table["flagged"] <= 1 - min_coverage)]
    if len(ok):
        return float(ok.iloc[0]["threshold"])
    return float(table.sort_values("accuracy_of_accepted", ascending=False).iloc[0]["threshold"])


def train(a):
    tr = load_english(a.with_311)
    print(f"\nEnglish training rows: {len(tr):,}")
    X = cached_vectors(tr["text"].tolist())
    y = tr["category"].values

    if not CHECK_PATH.exists():
        raise SystemExit(f"Missing {CHECK_PATH} (the Arabic check set)")
    ck = pd.read_csv(CHECK_PATH).dropna(subset=["text", "category"]).reset_index(drop=True)
    ck["text"] = ck["text"].map(clean_text)
    Xc = cached_vectors(ck["text"].tolist())

    mean_en, mean_ar = X.mean(0), Xc.mean(0)  # centering: text only, no labels
    Xn, Xcn = _unit(X - mean_en), _unit(Xc - mean_ar)
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
    clf.fit(Xn, y)

    proba = clf.predict_proba(Xcn)
    pred = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    correct = pred == ck["category"].values

    f1_all = f1_score(ck["category"], pred, average="macro")
    print(f"\n=== Arabic/Egyptian check set ({len(ck)} rows), no Arabic labels used in training ===")
    print(f"category macro-F1 overall: {f1_all:.3f}   (target on the real test set: 0.85)")
    metrics = {"f1_check_overall": f1_all}
    if "source" in ck.columns:
        for src, g in ck.groupby("source"):
            f = f1_score(g["category"], pred[g.index], average="macro")
            metrics[f"f1_{src}"] = f
            print(f"  {src}: macro-F1 {f:.3f} ({len(g)} rows)")

    # ---- urgency (second priority; labels are noisy) ----
    urg_clf = None
    yu = tr["urgency"].fillna("").astype(str).str.lower().str.strip().values
    has_u = np.isin(yu, URGENCY_LEVELS)
    if len(set(yu[has_u])) >= 2:
        urg_clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
        urg_clf.fit(Xn[has_u], yu[has_u])
        if "urgency" in ck.columns:
            true_u = ck["urgency"].astype(str).str.lower().str.strip()
            pu = urg_clf.predict(Xcn)
            fu = f1_score(true_u, pu, labels=URGENCY_LEVELS, average="macro", zero_division=0)
            metrics["f1_urgency_check"] = fu
            print(f"\nurgency macro-F1 on the check set: {fu:.3f}  (accuracy {(true_u.values == pu).mean():.3f})")
            print(classification_report(true_u, pu, labels=URGENCY_LEVELS, zero_division=0))
    else:
        print("\nNo usable urgency labels in the training data: urgency model skipped")

    table = threshold_table(conf, correct)
    thr = pick_threshold(table, a.target_acc)
    print(f"\nReview threshold table (tuned on the check set, so provisional):\n{table.to_string(index=False)}")
    print(f"\nChosen threshold: {thr:.2f}  (smallest value where accepted predictions are >= {a.target_acc:.0%} correct, "
          f"keeping at least half the requests automatic)")
    metrics["threshold"] = thr
    metrics["flagged_at_threshold"] = float((conf < thr).mean())

    art = {"clf": clf, "urg_clf": urg_clf, "model": EMB_MODEL, "host": HOST, "prefix": PREFIX, "mean_en": mean_en, "mean_ar": mean_ar,
           "threshold": thr, "C": C, "n_train": len(tr), "seed": SEED,
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    if a.mlflow:
        try:
            import mlflow
        except ImportError:
            print("mlflow is not installed (pip install mlflow); skipping logging")
            return
        mlflow.set_experiment("category-classifier")
        with mlflow.start_run(run_name="qwen3-emb4b-prefix-center"):
            mlflow.log_params({"embedding_model": EMB_MODEL, "prefix": PREFIX, "C": C, "centering": "per-language mean",
                               "with_311": a.with_311, "n_train": len(tr), "target_acc": a.target_acc})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(MODEL_PATH))
        print("Logged to MLflow (run: mlflow ui)")


# ---------- inference ----------
class CategoryClassifier:
    def __init__(self, art):
        self.art = art
        self.threshold = art["threshold"]

    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(joblib.load(path))

    def predict(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        cleaned = [clean_text(t) for t in texts]
        V = _unit(embed([self.art["prefix"] + t for t in cleaned], self.art["model"]))
        langs = [detect_lang(t) for t in cleaned]
        for i, lang in enumerate(langs):
            V[i] = V[i] - (self.art["mean_ar"] if lang == "ar" else self.art["mean_en"])
        Vn = _unit(V)
        proba = self.art["clf"].predict_proba(Vn)
        classes = self.art["clf"].classes_
        urg = self.art.get("urg_clf")
        uproba = urg.predict_proba(Vn) if urg is not None else None
        out = []
        for i, (p, lang) in enumerate(zip(proba, langs)):
            conf = float(p.max())
            row = {"category": str(classes[p.argmax()]), "confidence": round(conf, 3),
                   "needs_review": conf < self.threshold, "language": lang}
            if uproba is not None:
                row["urgency"] = str(urg.classes_[uproba[i].argmax()])
                row["urgency_confidence"] = round(float(uproba[i].max()), 3)
            out.append(row)
        return out[0] if single else out


def predict_cli(a):
    clf = CategoryClassifier.load()
    for text, r in zip(a.texts, clf.predict(a.texts)):
        flag = "REVIEW" if r["needs_review"] else "auto"
        urg = f" urgency={r['urgency']}" if "urgency" in r else ""
        print(f"[{r['category']:<12} {r['confidence']:.2f} {flag}{urg}] ({r['language']}) {text}")


def _norm_cat(c):
    return str(c).strip().lower().replace("_", " ").replace("-", " ")


def _key(t):
    return "".join(str(t).split()).lower()


def load_testset(classes):
    """Reads data/test/testset_*.csv (written by build_testset.py); falls back to test_handwritten.csv."""
    folder = ROOT / "data" / "test"
    frames = []
    for p in sorted(folder.glob("testset_*.csv")):
        df = pd.read_csv(p, encoding="utf-8-sig")
        df["set"] = p.stem.replace("testset_", "")
        frames.append(df)
    if not frames and TEST_PATH.exists():
        df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
        df["set"] = df["language"] if "language" in df.columns else "all"
        frames.append(df)
    if not frames:
        raise SystemExit(f"No test set found in {folder} (run ml\\build_testset.py first).")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "category"]).reset_index(drop=True)
    lookup = {_norm_cat(c): c for c in classes}  # test says pest_control, training says "pest control"
    unknown = sorted({c for c in df["category"].map(_norm_cat) if c not in lookup})
    if unknown:
        raise SystemExit(f"Test labels the model does not know: {unknown}")
    df["category"] = df["category"].map(lambda c: lookup[_norm_cat(c)])
    return df


def leaked_rows(df):
    """True for test texts that also appear in any training pool or the Arabic check set."""
    seen = set()
    for p in list(PROC.glob("*pool*.csv")) + [CHECK_PATH]:
        if p.exists():
            d = pd.read_csv(p)
            if "text" in d.columns:
                seen |= {_key(t) for t in d["text"].dropna()}
    return df["text"].map(_key).isin(seen)


def test_cli(a):
    clf = CategoryClassifier.load()
    df = load_testset(clf.art["clf"].classes_)
    leak = leaked_rows(df)
    if leak.any():
        print(f"WARNING: {int(leak.sum())} test texts also appear in training or check data. Remove them from the test set:")
        for t in df.loc[leak, "text"].head(10):
            print("  ", t[:80])
    res = pd.DataFrame(clf.predict(df["text"].tolist()))
    df["pred"], df["conf"], df["review"] = res["category"], res["confidence"], res["needs_review"]
    ok = df["pred"] == df["category"]
    print(f"\n=== HAND-WRITTEN TEST SET ({len(df)} rows) ===")
    f1 = f1_score(df["category"], df["pred"], average="macro")
    print(f"category macro-F1: {f1:.3f}   target 0.85   -> {'PASS' if f1 >= 0.85 else 'NOT YET'}")
    for name, g in df.groupby("set"):
        print(f"  {name}: macro-F1 {f1_score(g['category'], g['pred'], average='macro'):.3f} ({len(g)} rows)")
    print(f"flagged for review at threshold {clf.threshold:.2f}: {df['review'].mean():.1%}")
    if "urgency" in res.columns and "urgency" in df.columns:
        true_u = df["urgency"].astype(str).str.lower().str.strip().replace({"emergency": "high"})  # 4 levels -> 3
        fu = f1_score(true_u, res["urgency"], labels=URGENCY_LEVELS, average="macro", zero_division=0)
        print(f"urgency macro-F1 (emergency counted as high): {fu:.3f}   accuracy {(true_u.values == res['urgency'].values).mean():.3f}")
        print("  note: urgency training labels follow the generator's rules, not the labeling guide, so expect a gap")
    auto = ~df["review"]
    if auto.any():
        print(f"accuracy on the requests it handled automatically: {ok[auto].mean():.3f}")
    labels = sorted(df["category"].unique())
    print(pd.DataFrame(confusion_matrix(df["category"], df["pred"], labels=labels), index=labels, columns=labels))
    out = ROOT / "data" / "test" / "test_errors.csv"
    df[~ok].sort_values("conf", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{(~ok).sum()} wrong predictions saved to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--mlflow", action="store_true")
    t.add_argument("--with-311", action="store_true")
    t.add_argument("--target-acc", type=float, default=0.95, help="accuracy wanted on auto-handled requests")
    p = sub.add_parser("predict")
    p.add_argument("texts", nargs="+")
    sub.add_parser("test")
    a = ap.parse_args()
    {"train": train, "predict": predict_cli, "test": test_cli}[a.cmd](a)


if __name__ == "__main__":
    main()