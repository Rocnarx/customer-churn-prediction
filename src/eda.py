import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET_COLUMN
from src.data import (
    download_data,
    load_raw_data,
    clean_data,
    save_processed_data,
    ensure_directories,
)


def save_target_distribution(df: pd.DataFrame) -> None:
    """Save target distribution chart."""
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure()
    target_counts.plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"], rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close()


def save_churn_by_contract(df: pd.DataFrame) -> None:
    """Save average churn rate by contract type."""
    if "Contract" not in df.columns:
        return

    churn_by_contract = (
        df.groupby("Contract")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()
    churn_by_contract.plot(kind="bar")
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Average Churn Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png")
    plt.close()


def save_monthly_charges_by_churn(df: pd.DataFrame) -> None:
    """Save boxplot of MonthlyCharges grouped by churn."""
    if "MonthlyCharges" not in df.columns:
        return

    plt.figure()
    df.boxplot(column="MonthlyCharges", by=TARGET_COLUMN)
    plt.title("Monthly Charges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.xticks(ticks=[1, 2], labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_charges_by_churn.png")
    plt.close()


def save_eda_summary(df: pd.DataFrame) -> None:
    """Save a simple text summary of the dataset."""
    summary_path = REPORTS_DIR / "eda_summary.txt"

    missing_values = df.isna().sum().sort_values(ascending=False)
    target_distribution = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Customer Churn Prediction - EDA Summary\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Dataset shape: {df.shape}\n\n")

        file.write("Target distribution:\n")
        file.write(target_distribution.to_string())
        file.write("\n\n")

        file.write("Missing values:\n")
        file.write(missing_values.to_string())
        file.write("\n\n")

        file.write("Numeric summary:\n")
        file.write(df.describe().to_string())

    print(f"EDA summary saved to: {summary_path}")


def main() -> None:
    ensure_directories()
    download_data()

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)

    save_processed_data(clean_df)

    save_target_distribution(clean_df)
    save_churn_by_contract(clean_df)
    save_monthly_charges_by_churn(clean_df)
    save_eda_summary(clean_df)

    print("EDA completed. Figures saved in reports/figures/.")


if __name__ == "__main__":
    main()