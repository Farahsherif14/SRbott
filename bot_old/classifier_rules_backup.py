"""Category + urgency classifier used by app/flow.py (classify_node -> save_node).

Uses a trained model when one exists (settings.classifier_path, or models/classifier.joblib from train.py).
Until then it falls back to keyword rules, so confirmed requests are saved instead of being rejected with
"classification is not implemented yet"."""
import logging
import re
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)
REVIEW_THRESHOLD = 0.6

CANON = {"plumbing": "plumbing", "electricity": "electricity", "ac": "AC", "carpentry": "carpentry",
         "pest control": "pest control", "pest_control": "pest control", "general": "general"}
URGENCIES = {"low", "medium", "high", "emergency"}

_model = None
_loaded = False


def _normalize(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", s)
    s = re.sub("[╪Ñ╪ú╪ó]", "╪º", s)
    return s.replace("┘ë", "┘è").replace("╪⌐", "┘ç")


def _load():
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    try:
        import joblib
        for candidate in (settings.classifier_path, "models/classifier.joblib"):
            if candidate and Path(candidate).exists():
                _model = joblib.load(candidate)
                logger.info("Loaded classifier from %s", candidate)
                break
    except Exception as exc:
        logger.warning("Could not load trained classifier (%s); using keyword rules", type(exc).__name__)
        _model = None
    return _model


# ---------------- keyword fallback ----------------
CATEGORY_KW = {
    "plumbing": ["leak", "sink", "clog", "water", "pipe", "toilet", "drain", "faucet", "flood",
                 "┘à┘è╪⌐", "┘à┘è┘ç", "╪º┘ä╪¡┘ê╪╢", "╪¡┘ê╪╢", "┘à╪│╪»┘ê╪»", "╪¡┘å┘ü┘è┘ç", "┘à╪º╪│┘ê╪▒┘ç", "╪¬╪│╪▒┘è╪¿", "╪│╪¿╪º┘â┘ç"],
    "electricity": ["power", "electric", "socket", "bulb", "light", "outlet", "breaker", "wire", "panel", "fuse",
                    "╪º┘ä┘å┘ê╪▒", "┘å┘ê╪▒", "┘â┘ç╪▒╪¿╪º", "┘â┘ç╪▒╪¿╪º╪í", "┘ä┘à╪¿┘ç", "┘ü┘è╪┤┘ç", "┘é╪º╪╖╪╣"],
    "AC": ["ac", "cooling", "conditioner", "╪º┘ä╪¬┘â┘è┘è┘ü", "╪¬┘â┘è┘è┘ü", "┘à┘â┘è┘ü"],
    "carpentry": ["door", "closet", "cabinet", "wardrobe", "hinge", "window", "lock",
                  "╪º┘ä╪¿╪º╪¿", "╪¿╪º╪¿", "╪»┘ê┘ä╪º╪¿", "┘à┘ü╪╡┘ä┘ç", "╪┤╪¿╪º┘â", "┘å╪¼╪º╪▒"],
    "pest control": ["ants", "ant", "cockroach", "cockroaches", "rat", "mice", "mouse", "bees",
                     "mosquitoes", "pest", "┘å┘à┘ä", "╪╡╪▒╪º╪╡┘è╪▒", "┘ü╪º╪▒", "┘ü┘è╪▒╪º┘å", "╪¡╪┤╪▒╪º╪¬"],
}
EMERGENCY_KW = ["fire", "smoke", "spark", "sparks", "gas leak", "burning", "electric shock",
                "╪¡╪▒┘è┘é", "╪»╪«╪º┘å", "╪┤╪▒╪▒", "╪║╪º╪▓", "┘à╪º╪│ ┘â┘ç╪▒╪¿╪º╪ª┘è"]
HIGH_KW = ["flood", "pouring", "no power", "urgent", "emergency", "rat", "┘à┘ê┘ä╪╣", "┘ü┘è╪╢╪º┘å", "╪╣╪º╪¼┘ä", "╪╖┘ê╪º╪▒╪ª", "┘ü╪º╪▒"]
LOW_KW = ["not urgent", "whenever", "no rush", "minor", "┘à╪┤ ┘à╪│╪¬╪╣╪¼┘ä", "┘ü┘è ╪º┘è ┘ê┘é╪¬"]


def _has(norm: str, tokens: set, kw: str) -> bool:
    kw = _normalize(kw)
    if " " in kw:
        return kw in norm
    return kw in tokens or (len(kw) > 3 and any(tok.startswith(kw) for tok in tokens))


def _keyword_predict(text: str) -> dict:
    norm = _normalize(text)
    tokens = set(re.findall(r"\w+", norm))
    scores = {c: sum(_has(norm, tokens, k) for k in kws) for c, kws in CATEGORY_KW.items()}
    best = max(scores.values())
    if best == 0:
        category, cat_conf = "general", 0.30
    else:
        top = [c for c, v in scores.items() if v == best]
        category, cat_conf = top[0], (0.80 if len(top) == 1 else 0.50)
    if any(_has(norm, tokens, k) for k in EMERGENCY_KW):
        urgency = "emergency"
    elif any(_has(norm, tokens, k) for k in HIGH_KW):
        urgency = "high"
    elif any(_has(norm, tokens, k) for k in LOW_KW):
        urgency = "low"
    else:
        urgency = "medium"
    return {"category": category, "category_confidence": cat_conf, "urgency": urgency,
            "urgency_confidence": 0.60, "model_version": "keywords-v0"}


def _top(pipe, text: str):
    probs = pipe.predict_proba([text])[0]
    i = int(probs.argmax())
    return str(pipe.classes_[i]), float(probs[i])


def classify(text: str) -> dict:
    model = _load()
    out = None
    if model is not None:
        try:
            if isinstance(model, dict) and "category" in model:       # model saved by train.py
                cat, cc = _top(model["category"], text)
                if "urgency" in model:
                    urg, uc = _top(model["urgency"], text)
                else:
                    kw = _keyword_predict(text)
                    urg, uc = kw["urgency"], kw["urgency_confidence"]
                version = str(model.get("version", "tfidf-lr"))
            elif hasattr(model, "predict_proba"):                      # a single category-only pipeline
                cat, cc = _top(model, text)
                kw = _keyword_predict(text)
                urg, uc = kw["urgency"], kw["urgency_confidence"]
                version = "category-model"
            else:
                raise TypeError("unsupported model object")
            out = {"category": cat, "category_confidence": cc, "urgency": urg,
                   "urgency_confidence": uc, "model_version": version}
        except Exception as exc:
            logger.warning("Trained classifier failed (%s); using keyword rules", type(exc).__name__)
    if out is None:
        out = _keyword_predict(text)
    out["category"] = CANON.get(str(out["category"]).strip().lower(), "general")
    if out["urgency"] not in URGENCIES:
        out["urgency"] = "medium"
    out["confidence"] = float(out["category_confidence"])
    out["needs_review"] = bool(min(out["category_confidence"], out["urgency_confidence"]) < REVIEW_THRESHOLD)
    return out
