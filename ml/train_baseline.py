"""TF-IDF + logistic regression baseline for category and urgency.

Evaluates on validation by default. Add --test to also score the hand-written
test set (do this only at the project gates, never to tune the model).
"""
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from config import PROC, TEST, MODELS
from text_utils import clean_text

try:
    import mlflow
except ImportError:
    mlflow = None


def make_pipe():
    return Pipeline([
        # character n-grams cope with Arabic morphology and spelling variants
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also score the hand-written test set (gates only)")
    ap.add_argument("--threshold", type=float, default=0.6, help="below this confidence -> flag for review")
    a = ap.parse_args()

    train = pd.read_csv(PROC / "train.csv")
    val = pd.read_csv(PROC / "val.csv")
    MODELS.mkdir(exist_ok=True)
    if mlflow:
        mlflow.start_run(run_name="tfidf_lr_baseline")
        mlflow.log_param("threshold", a.threshold)

    for target in ("category", "urgency"):
        pipe = make_pipe().fit(train["text"].map(clean_text), train[target])
        vx = val["text"].map(clean_text)
        pred = pipe.predict(vx)
        conf = pipe.predict_proba(vx).max(axis=1)
        f1 = f1_score(val[target], pred, average="macro")
        print(f"\n=== {target} | val macro-F1 = {f1:.3f} | flagged for review: {(conf < a.threshold).mean():.1%}")
        print(classification_report(val[target], pred, zero_division=0))
        print(pd.DataFrame(confusion_matrix(val[target], pred, labels=pipe.classes_),
                           index=pipe.classes_, columns=pipe.classes_))
        if mlflow:
            mlflow.log_metric(f"val_macro_f1_{target}", f1)

        if a.test:
            test = pd.read_csv(TEST / "test_handwritten.csv")
            tpred = pipe.predict(test["text"].map(clean_text))
            tf1 = f1_score(test[target], tpred, average="macro")
            print(f"--- {target} | TEST macro-F1 = {tf1:.3f}")
            print(classification_report(test[target], tpred, zero_division=0))
            if mlflow:
                mlflow.log_metric(f"test_macro_f1_{target}", tf1)

        joblib.dump(pipe, MODELS / f"baseline_{target}.joblib")
    if mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()