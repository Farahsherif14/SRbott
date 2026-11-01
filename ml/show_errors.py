"""Show where the category model goes wrong on validation.
    python -u ml/show_errors.py
Run after ml/train_embeddings.py (it loads ml/category_embed.joblib).
"""
import sys

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

saved = joblib.load("ml/category_embed.joblib")
clf, name = saved["clf"], saved["model"]
val = pd.read_csv("data/processed/val.csv").dropna(subset=["text"])
val = val[val["source"] != "nyc311"].reset_index(drop=True)

enc = SentenceTransformer(name)
X = enc.encode(list(val["text"]), normalize_embeddings=True, show_progress_bar=False)
probs = clf.predict_proba(X)
pred = clf.classes_[probs.argmax(axis=1)]
conf = probs.max(axis=1)

labels = list(clf.classes_)
print("Confusion matrix (rows = true, columns = predicted)\n")
cm = pd.DataFrame(confusion_matrix(val["category"], pred, labels=labels),
                  index=labels, columns=labels)
print(cm.to_string())

wrong = val[pred != val["category"]].copy()
wrong["predicted"] = pred[pred != val["category"]]
wrong["conf"] = conf[pred != val["category"]].round(2)
print(f"\n{len(wrong)} wrong of {len(val)}:\n")
for _, r in wrong.sort_values("category").iterrows():
    print(f"true={r['category']:12s} pred={r['predicted']:12s} conf={r['conf']}  [{r['language']}] {r['text']}")
    