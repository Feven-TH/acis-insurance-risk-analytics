from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_DIR / "data" / "clean" / "insurance_claims_cleaned.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
VISUALS_DIR = PROJECT_DIR / "visuals"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)
MIN_GROUP_N = 30
ALPHA = 0.05