import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")  # tashkeel + tatweel
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s\-()]{7,}\d(?!\w)")
_ZW_SPACE = re.compile(r"[\u200b\u2060\ufeff]")
_ZW_LATIN = re.compile(r"(?<=[A-Za-z'])[\u200c\u200d](?=[A-Za-z])")
_ZW_DROP = re.compile(r"[\u200c\u200d\u200e\u200f\u00ad]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_AR_CHARS = re.compile(r"[\u0600-\u06FF]")


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    return text.translate(_AR_DIGITS)


def clean_text(text) -> str:
    """Normalize unicode, mask contact details, normalize Arabic, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZW_SPACE.sub(" ", text)  # invisible separators some models emit between words
    text = _ZW_LATIN.sub(" ", text)  # joiner between English letters acted as a word break
    text = _ZW_DROP.sub("", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    text = normalize_arabic(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    ar = sum(1 for c in letters if _AR_CHARS.match(c))
    return "ar" if ar / len(letters) > 0.5 else "en"