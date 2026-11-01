from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TEST = ROOT / "data" / "test"
MODELS = ROOT / "models"

CATEGORIES = ["plumbing", "electricity", "ac", "carpentry", "pest control", "general"]
URGENCY = ["low", "medium", "high"]

SEED = 42
MAX_PER_COMBO = 300      # cap rows per (complaint type, descriptor) so big classes don't dominate
VAL_FRACTION_FOLDS = 7   # 1 of 7 folds (~14%) becomes validation