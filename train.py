from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")

RANDOM_STATE = 42
DATA_PATH = Path("data") / "heart_disease_uci.csv"
OUTPUT_DIR = Path("outputs")
MODEL_PATH = Path("model.pkl")
TUNED_MODEL_PATH = Path("tuned_model.pkl")
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve.png"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"
SHAP_BEESWARM_PATH = OUTPUT_DIR / "shap_beeswarm.png"
TARGET_COLUMN = "num"
DROP_COLUMNS = ["id", "dataset", TARGET_COLUMN]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the UCI heart disease dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def create_binary_target(df: pd.DataFrame) -> pd.Series:
    """Convert UCI severity codes into a binary risk target."""
    return (df[TARGET_COLUMN] > 0).astype(int)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep patient health variables and remove identifiers/source metadata."""
    return df.drop(columns=DROP_COLUMNS)


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def build_model_pipelines(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, Pipeline]:
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, scale_numeric=True),
            ),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, scale_numeric=False),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return logistic_pipeline, random_forest_pipeline


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def run_cross_validation(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    scoring_metrics = {"accuracy": "accuracy", "recall": "recall", "f1": "f1"}

    for model_name, model in models.items():
        row = {"model": model_name}
        for metric_name, scoring in scoring_metrics.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            row[f"mean_{metric_name}"] = scores.mean()
            row[f"std_{metric_name}"] = scores.std()
        rows.append(row)

    return pd.DataFrame(rows)


def tune_random_forest(
    random_forest_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [3, 5],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search


def plot_roc_curve(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = ROC_CURVE_PATH,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 7))

    auc_scores = {}
    for model_name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        auc_scores[model_name] = auc_score
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.5,
            label=f"{model_name} (AUC = {auc_score:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random baseline")
    plt.title("ROC Curve: Heart Disease Risk Prediction", fontsize=16, weight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate / Recall", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return auc_scores


def get_transformed_feature_matrix(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def positive_class_shap_values(shap_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(shap_values, list):
        return shap_values[1]
    if getattr(shap_values, "ndim", 0) == 3:
        return shap_values[:, :, 1]
    return shap_values


def save_shap_plots(
    tuned_random_forest: Pipeline,
    X_test: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_transformed = get_transformed_feature_matrix(tuned_random_forest, X_test)
    sample_size = min(150, len(X_transformed))
    X_sample = X_transformed.sample(sample_size, random_state=RANDOM_STATE)

    forest_model = tuned_random_forest.named_steps["model"]
    explainer = shap.TreeExplainer(forest_model)
    shap_values = positive_class_shap_values(explainer.shap_values(X_sample))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=12, show=False)
    plt.title("SHAP Summary: Top Drivers of Predicted Heart Disease Risk", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, max_display=12, show=False)
    plt.title("SHAP Beeswarm: Feature Impact Direction and Magnitude", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    return shap_importance


def threshold_analysis(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    probabilities = model.predict_proba(X_test)[:, 1]
    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    y = create_binary_target(df)
    X = prepare_features(df)
    numeric_features, categorical_features = split_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logistic_pipeline, random_forest_pipeline = build_model_pipelines(
        numeric_features,
        categorical_features,
    )

    logistic_pipeline.fit(X_train, y_train)
    random_forest_pipeline.fit(X_train, y_train)
    tuned_search = tune_random_forest(random_forest_pipeline, X_train, y_train)
    tuned_random_forest = tuned_search.best_estimator_

    evaluation_results = pd.DataFrame(
        [
            evaluate_model("Logistic Regression", logistic_pipeline, X_test, y_test),
            evaluate_model("Random Forest", random_forest_pipeline, X_test, y_test),
            evaluate_model("Tuned Random Forest", tuned_random_forest, X_test, y_test),
        ]
    )

    cv_results = run_cross_validation(
        {
            "Logistic Regression": logistic_pipeline,
            "Random Forest": random_forest_pipeline,
        },
        X,
        y,
    )

    auc_scores = plot_roc_curve(
        {
            "Logistic Regression": logistic_pipeline,
            "Random Forest": random_forest_pipeline,
            "Tuned Random Forest": tuned_random_forest,
        },
        X_test,
        y_test,
    )
    shap_importance = save_shap_plots(tuned_random_forest, X_test)
    threshold_results = threshold_analysis(tuned_random_forest, X_test, y_test)

    model_package = {
        "model": random_forest_pipeline,
        "model_name": "Random Forest",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_definition": "1 if num > 0, else 0",
        "random_state": RANDOM_STATE,
    }
    tuned_model_package = {
        "model": tuned_random_forest,
        "model_name": "Tuned Random Forest",
        "best_params": tuned_search.best_params_,
        "best_cv_f1": tuned_search.best_score_,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_definition": "1 if num > 0, else 0",
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_package, MODEL_PATH)
    joblib.dump(tuned_model_package, TUNED_MODEL_PATH)

    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    print("\nModel evaluation on held-out test set:")
    print(evaluation_results.round(3).to_string(index=False))

    print("\nClassification report: Logistic Regression")
    print(classification_report(y_test, logistic_pipeline.predict(X_test), target_names=["No disease", "Disease"]))

    print("Classification report: Tuned Random Forest")
    print(classification_report(y_test, tuned_random_forest.predict(X_test), target_names=["No disease", "Disease"]))

    print("\n5-fold cross-validation:")
    print(cv_results.round(3).to_string(index=False))

    print("\nRandom Forest tuning:")
    print(f"Best parameters: {tuned_search.best_params_}")
    print(f"Best CV F1 score: {tuned_search.best_score_:.3f}")

    print("\nROC-AUC scores:")
    for model_name, auc_score in auc_scores.items():
        print(f"{model_name}: {auc_score:.3f}")

    print("\nTuned Random Forest threshold analysis:")
    print(threshold_results.round(3).to_string(index=False))

    print("\nTop SHAP features:")
    print(shap_importance.head(8).round(4).to_string(index=False))

    print(f"\nSaved model package to {MODEL_PATH}")
    print(f"Saved tuned model package to {TUNED_MODEL_PATH}")
    print(f"Saved ROC curve to {ROC_CURVE_PATH}")
    print(f"Saved SHAP summary to {SHAP_SUMMARY_PATH}")
    print("Project pipeline executed successfully.")


if __name__ == "__main__":
    main()
