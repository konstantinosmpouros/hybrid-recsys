from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS_MODELS = ROOT / "artifacts" / "models"
ARTIFACTS_METRICS = ROOT / "artifacts" / "metrics"

RATING_SCALE = (0.5, 5.0)
RELEVANCE_THRESHOLD = 4.0
K_VALUES = [5, 10, 20]

RANDOM_STATE = 42
