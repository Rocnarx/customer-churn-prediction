import json
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
)
from src.data import (
    download_data,
    load_raw_data,
    clean_data,
    save_processed_data,
    get_features_and_target,
    ensure_directories,
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_models(X: pd.DataFrame) -> dict:
    """Build candidate models."""
    preprocessor = build_preprocessor(X)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def save_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save confusion matrix for the selected model."""
    y_pred = model.predict(X_test)

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close()


def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from Random Forest or absolute coefficients
    from Logistic Regression.
    """
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        importance_values = abs(estimator.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        raise ValueError("The selected model does not expose feature importance.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            importance_type: importance_values,
        }
    ).sort_values(by=importance_type, ascending=False)

    return importance_df


def save_feature_importance(model) -> None:
    """Save feature importance as CSV and chart."""
    importance_df = get_feature_importance(model)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    value_column = importance_df.columns[1]
    top_features = importance_df.head(15).sort_values(by=value_column)

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features[value_column])
    plt.title("Top 15 Feature Importance")
    plt.xlabel(value_column)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close()

    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")


def main() -> None:
    ensure_directories()
    download_data()

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df)

    X, y = get_features_and_target(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(X_train)

    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics

        print(f"{model_name} metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = models[best_model_name]

    output = {
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    joblib.dump(
        {
            "model": best_model,
            "metadata": output,
            "feature_columns": X.columns.tolist(),
        },
        MODEL_PATH,
    )

    save_confusion_matrix(best_model, X_test, y_test)
    save_feature_importance(best_model)

    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()