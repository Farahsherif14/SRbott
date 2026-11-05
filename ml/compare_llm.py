r"""ML classifier vs LLM zero-shot: accuracy, latency, cost per request.

Both systems classify the SAME hand-written test messages (data\test\testset_*.csv).
  ML  : your saved model (qwen3-embedding:4b embeddings + logistic regression, review flag)
  LLM : a local Ollama model asked to pick a category with no training examples (zero-shot)

Run from the project root, after the test set is built and the model is trained:
  python ml\compare_llm.py                                   # qwen3:4b, no cost figure
  python ml\compare_llm.py --price-in 1.0 --price-out 5.0    # + hosted cost estimate (USD per 1M tokens)
  python ml\compare_llm.py --limit 20                        # quick trial on 20 rows

Notes on fairness:
  * Each system runs in its own phase (all ML first, then all LLM) after a warm-up call, so a small GPU
    is not swapping the two models in and out, which would distort latency.
  * Latency is end to end per message, one message at a time, on THIS machine.
  * Cost: your classifier is self-hosted (no per-request fee). For the LLM, tokens are counted from Ollama
    and priced with the numbers YOU pass in. Use the current price of the hosted model you would really use;
    the script has no built-in prices.
  * Like the real test, this should be run only at gates, not while tuning.

Writes data\test\comparison_results.csv (per message) and data\test\comparison_summary.md (the table).
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from category_model import CategoryClassifier, ROOT

SYSTEM = """You sort maintenance requests from residents of a residential compound in Egypt (English, Arabic or Egyptian Arabic) into exactly one category.

Categories:
- plumbing: leaks, pipes, drains, toilets, taps, hot water, water pressure
- electricity: power cuts, sockets, switches, breaker, wiring, lights
- ac: air conditioner, cooling or heating unit, thermostat, remote
- carpentry: doors, windows, locks, cabinets, wardrobes, wooden floors inside the apartment
- pest control: insects, rodents, other pests
- general: paint, cracks, cleaning, elevator, garbage, intercom, gates, common areas, anything unclear

Reply with JSON only: {"category": "<one of: plumbing, electricity, ac, carpentry, pest control, general>"}"""


def norm(c):
    return str(c).strip().lower().replace("_", " ").replace("-", " ")


def post(host, path, payload, timeout=600):
    req = urllib.request.Request(f"{host}{path}", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_test(classes):
    folder = ROOT / "data" / "test"
    frames = []
    for p in sorted(folder.glob("testset_*.csv")):
        d = pd.read_csv(p, encoding="utf-8-sig")
        d["set"] = p.stem.replace("testset_", "")
        frames.append(d)
    if not frames:
        raise SystemExit(f"No testset_*.csv in {folder}. Run ml\\build_testset.py first.")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "category"]).reset_index(drop=True)
    lookup = {norm(c): c for c in classes}
    bad = sorted({c for c in df["category"].map(norm) if c not in lookup})
    if bad:
        raise SystemExit(f"Unknown test labels: {bad}")
    df["category"] = df["category"].map(lambda c: lookup[norm(c)])
    return df


def parse_llm(raw, classes):
    """Return one of the classes, or 'INVALID'."""
    lookup = {norm(c): c for c in classes}
    try:
        val = norm(json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("category", ""))
        if val in lookup:
            return lookup[val]
    except Exception:
        pass
    found = [lookup[k] for k in lookup if re.search(rf"\b{re.escape(k)}\b", norm(raw))]
    return found[0] if len(found) == 1 else "INVALID"


def ask_llm(host, model, text):
    body = {"model": model, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 40},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Request: {text}"}]}
    t0 = time.perf_counter()
    r = post(host, "/api/chat", body)
    dt = time.perf_counter() - t0
    return r["message"]["content"], dt, r.get("prompt_eval_count", 0), r.get("eval_count", 0)


def pct(x, q):
    return float(np.percentile(x, q))


def score(y, pred, classes):
    return f1_score(y, pred, labels=list(classes), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="qwen3:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (trial runs)")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens of the hosted LLM you would use")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    a = ap.parse_args()

    clf = CategoryClassifier.load()
    classes = list(clf.art["clf"].classes_)
    df = load_test(classes)
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0).reset_index(drop=True)
        print(f"TRIAL RUN on {len(df)} rows: numbers are not your final result")
    texts = df["text"].tolist()
    print(f"Test messages: {len(df)}  ({df['set'].value_counts().to_dict()})")

    # ---- phase 1: ML classifier, one message at a time ----
    print("\nPhase 1: ML classifier")
    clf.predict(texts[0])  # warm-up, not timed
    ml_pred, ml_conf, ml_review, ml_ms = [], [], [], []
    for i, t in enumerate(texts, 1):
        t0 = time.perf_counter()
        r = clf.predict(t)
        ml_ms.append((time.perf_counter() - t0) * 1000)
        ml_pred.append(r["category"]); ml_conf.append(r["confidence"]); ml_review.append(r["needs_review"])
        if i % 25 == 0 or i == len(texts):
            print(f"  {i}/{len(texts)}")

    # ---- phase 2: LLM zero-shot ----
    print(f"\nPhase 2: {a.llm} zero-shot")
    _, _, warm_in, _ = ask_llm(a.host, a.llm, texts[0])  # warm-up (loads the model), not timed
    llm_pred, llm_ms, tok_in, tok_out, raws = [], [], [], [], []
    for i, t in enumerate(texts, 1):
        raw, dt, ti, to = ask_llm(a.host, a.llm, t)
        llm_pred.append(parse_llm(raw, classes)); llm_ms.append(dt * 1000)
        tok_in.append(ti); tok_out.append(to); raws.append(raw)
        if i % 25 == 0 or i == len(texts):
            print(f"  {i}/{len(texts)}")

    df["ml_pred"], df["ml_conf"], df["ml_review"], df["ml_ms"] = ml_pred, ml_conf, ml_review, ml_ms
    df["llm_pred"], df["llm_ms"], df["llm_in"], df["llm_out"], df["llm_raw"] = llm_pred, llm_ms, tok_in, tok_out, raws
    out_csv = ROOT / "data" / "test" / "comparison_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    y = df["category"].values
    ml_ok = df["ml_pred"].values == y
    llm_ok = df["llm_pred"].values == y
    auto = ~df["ml_review"].values
    in_tok = warm_in or (statistics.median(tok_in) if tok_in else 0)  # warm-up call = full uncached prompt
    out_tok = statistics.mean(tok_out) if tok_out else 0

    lines = []
    lines.append(f"Comparison on {len(df)} hand-written test messages ({', '.join(f'{k}={v}' for k, v in df['set'].value_counts().items())})\n")
    lines.append("| | ML classifier | LLM zero-shot (" + a.llm + ") |")
    lines.append("|---|---|---|")
    lines.append(f"| Category accuracy | {ml_ok.mean():.3f} | {llm_ok.mean():.3f} |")
    lines.append(f"| Category macro-F1 | {score(y, df['ml_pred'], classes):.3f} | {score(y, df['llm_pred'], classes):.3f} |")
    for name, g in df.groupby("set"):
        lines.append(f"| macro-F1, {name} | {score(g['category'], g['ml_pred'], classes):.3f} | {score(g['category'], g['llm_pred'], classes):.3f} |")
    lines.append(f"| Unparseable answers | 0 | {(df['llm_pred'] == 'INVALID').mean():.1%} |")
    lines.append(f"| Median latency per message | {statistics.median(ml_ms):.0f} ms | {statistics.median(llm_ms):.0f} ms |")
    lines.append(f"| 95th percentile latency | {pct(ml_ms, 95):.0f} ms | {pct(llm_ms, 95):.0f} ms |")
    lines.append(f"| Flagged for review (confidence < {clf.threshold:.2f}) | {(~auto).mean():.1%} | not available |")
    if auto.any():
        lines.append(f"| Accuracy on auto-handled requests | {ml_ok[auto].mean():.3f} | not available |")
    if a.price_in is not None and a.price_out is not None:
        cost = (in_tok * a.price_in + out_tok * a.price_out) / 1e6
        lines.append(f"| Cost per request | $0 per request (self-hosted) | ${cost:.6f} (about {in_tok:.0f} in + {out_tok:.0f} out tokens at your prices) |")
        lines.append(f"| Cost per 10,000 requests | $0 | ${cost * 10000:.2f} |")
    else:
        lines.append(f"| Tokens per request | n/a | about {in_tok:.0f} in + {out_tok:.0f} out (pass --price-in/--price-out for a cost) |")
    lines.append("\nNotes: latency is end to end on this machine, one message at a time. The ML classifier's cost is zero per request "
                 "because the embedding model and classifier run locally; hardware and electricity are not counted. LLM cost is an "
                 "estimate from token counts and the prices passed to the script.")
    summary = "\n".join(lines)
    print("\n" + summary)
    (ROOT / "data" / "test" / "comparison_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"\nPer-message results: {out_csv}")


if __name__ == "__main__":
    main()