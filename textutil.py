import re

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def normalize_text(s: str) -> str:
    """Lowercase and unify common Arabic spelling variants (picklable, used by the model)."""
    s = (s or "").lower()
    s = _TASHKEEL.sub("", s)
    s = re.sub("[إأآ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه")


def has_arabic(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", s or ""))