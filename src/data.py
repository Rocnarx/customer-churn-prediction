import requests
import pandas as pd

from src.config import (
    DATA_URL,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    ID_COLUMNS,
)


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def download_data(force: bool = False) -> None:
    """
    Download the Telco Customer Churn dataset if it is not already available.

    Parameters
    ----------
    force : bool
        If True, download the file again even if it already exists.
    """
    ensure_directories()

    if RAW_DATA_PATH.exists() and not force:
        print(f"Dataset already exists at: {RAW_DATA_PATH}")
        return

    print("Downloading dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    RAW_DATA_PATH.write_bytes(response.content)
    print(f"Dataset saved to: {RAW_DATA_PATH}")


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    if not RAW_DATA_PATH.exists():
        download_data()

    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.

    Steps:
    - Strip column names.
    - Convert TotalCharges to numeric.
    - Convert target column from Yes/No to 1/0.
    - Remove duplicate rows.
    """
    df = df.copy()

    df.columns = df.columns.str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    df = df.drop_duplicates()

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset."""
    ensure_directories()
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")


def get_features_and_target(df: pd.DataFrame):
    """Split a cleaned dataframe into features X and target y."""
    columns_to_drop = [TARGET_COLUMN]

    for column in ID_COLUMNS:
        if column in df.columns:
            columns_to_drop.append(column)

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN].astype(int)

    return X, y