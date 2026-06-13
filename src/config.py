from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telco_churn_clean.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]

RANDOM_STATE = 42
TEST_SIZE = 0.20