"""Meaning-based classifier: multilingual sentence embeddings + logistic regression.
    python ml/train_embeddings.py
"""
import sys

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.stdout.reconfigure(encoding="utf-8")  # so Arabic prints on Windows

MODEL = "paraphrase-multilingual-mpnet-base-v2" # swap for ...-mpnet-base-v2 for more accuracy
TRAIN, VAL, TEST = "data/processed/train.csv", "data/processed/val.csv", "data/test/test_set.csv"
EXCLUDE_NYC311 = True   # 311 sentences contain the category word; keep them out of training

DEMO = [  # none of these contain an obvious category keyword
    "The room stays hot even though the machine on the wall is running",
    "Water is coming through the ceiling of my bathroom",
    "Half the apartment went dark and the switch keeps jumping back",
    "I keep seeing little brown insects in the kitchen at night",
    "The bedroom won't close properly anymore",
    "الاوضة مبتبردش رغم ان الجهاز شغال",
    "في مية نازلة من السقف في الحمام",
]

encoder = SentenceTransformer(MODEL)


def load(path):
    return pd.read_csv(path).dropna(subset=["text"])


def embed(texts):
    return encoder.encode(list(texts), batch_size=32, normalize_embeddings=True,
                          show_progress_bar=True)


train, val = load(TRAIN), load(VAL)
if EXCLUDE_NYC311:
    train = train[train["source"] != "nyc311"]
    val = val[val["source"] != "nyc311"]
print(f"training rows: {len(train)}")
try:
    test = load(TEST)
except FileNotFoundError:
    test = None
    print("No test set yet - showing validation only.")

X_train, X_val = embed(train["text"]), embed(val["text"])
X_test = embed(test["text"]) if test is not None else None

for target in ("category", "urgency"):
    if target not in train.columns:
        continue
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X_train, train[target])

    for name, X, df in (("VAL", X_val, val), ("TEST", X_test, test)):
        if df is None or target not in df.columns:
            continue
        pred = clf.predict(X)
        print(f"\n===== {target.upper()} on {name} =====")
        print("macro F1:", round(f1_score(df[target], pred, average="macro"), 3))
        print(classification_report(df[target], pred, zero_division=0))

    if target == "category":
        probs = clf.predict_proba(embed(DEMO))
        print("\n--- keyword-free demo sentences ---")
        for text, p in zip(DEMO, probs):
            i = p.argmax()
            print(f"{clf.classes_[i]:13s} {p[i]:.2f}  {text}")

    joblib.dump({"model": MODEL, "clf": clf}, f"ml/{target}_embed.joblib")