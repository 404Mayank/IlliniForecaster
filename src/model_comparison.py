from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


TERM_ORDER = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4,
}

GRADE_WEIGHTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.67,
    "B+": 3.33,
    "B": 3.0,
    "B-": 2.67,
    "C+": 2.33,
    "C": 2.0,
    "C-": 1.67,
    "D+": 1.33,
    "D": 1.0,
    "D-": 0.67,
    "F": 0.0,
}


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.replace(0, np.nan)
    return num / den_safe


def compute_target_average_gpa(df: pd.DataFrame) -> pd.Series:
    weighted_points = pd.Series(0.0, index=df.index)
    graded_students = pd.Series(0.0, index=df.index)

    for grade_col, weight in GRADE_WEIGHTS.items():
        weighted_points += df[grade_col] * weight
        graded_students += df[grade_col]

    return _safe_divide(weighted_points, graded_students)


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["term_order"] = out["Term"].map(TERM_ORDER)

    if out["term_order"].isna().any():
        bad_terms = sorted(out.loc[out["term_order"].isna(), "Term"].unique())
        raise ValueError(f"Unexpected term values: {bad_terms}")

    out["time_key"] = out["Year"].astype(int) * 10 + out["term_order"].astype(int)
    unique_time_keys = np.sort(out["time_key"].unique())
    mapping = {time_key: i for i, time_key in enumerate(unique_time_keys)}
    out["time_index"] = out["time_key"].map(mapping).astype(int)
    return out


def _global_historical_mean(df: pd.DataFrame, target_col: str) -> pd.Series:
    by_term = (
        df.groupby("time_index", as_index=False)[target_col]
        .agg(term_sum="sum", term_count="count")
        .sort_values("time_index")
    )
    past_sum = by_term["term_sum"].cumsum() - by_term["term_sum"]
    past_count = by_term["term_count"].cumsum() - by_term["term_count"]
    by_term["global_hist_mean"] = _safe_divide(past_sum, past_count)

    merged = df[["time_index"]].merge(
        by_term[["time_index", "global_hist_mean"]],
        on="time_index",
        how="left",
    )
    return merged["global_hist_mean"]


def _group_historical_mean(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    output_col: str,
) -> pd.Series:
    group_term_stats = (
        df.groupby([group_col, "time_index"], as_index=False)[target_col]
        .agg(group_sum="sum", group_count="count")
        .sort_values([group_col, "time_index"])
    )

    cumsum_sum = group_term_stats.groupby(group_col)["group_sum"].cumsum()
    cumsum_count = group_term_stats.groupby(group_col)["group_count"].cumsum()
    past_sum = cumsum_sum - group_term_stats["group_sum"]
    past_count = cumsum_count - group_term_stats["group_count"]

    group_term_stats[output_col] = _safe_divide(past_sum, past_count)

    merged = df[[group_col, "time_index"]].merge(
        group_term_stats[[group_col, "time_index", output_col]],
        on=[group_col, "time_index"],
        how="left",
    )
    return merged[output_col]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_index(df)
    out["avg_gpa"] = compute_target_average_gpa(out)

    out = out.dropna(subset=["avg_gpa"]).copy()

    out["class_size"] = out["Students"].astype(float)
    out["course_level"] = (
        (pd.to_numeric(out["Number"], errors="coerce").fillna(0) // 100) * 100
    ).astype(int)

    out["global_hist_gpa"] = _global_historical_mean(out, target_col="avg_gpa")

    out["subject_hist_gpa"] = _group_historical_mean(
        out,
        group_col="Subject",
        target_col="avg_gpa",
        output_col="subject_hist_gpa",
    )

    out["instructor_hist_gpa"] = _group_historical_mean(
        out,
        group_col="Primary Instructor",
        target_col="avg_gpa",
        output_col="instructor_hist_gpa",
    )

    overall_mean = out["avg_gpa"].mean()
    out["subject_hist_gpa"] = out["subject_hist_gpa"].fillna(out["global_hist_gpa"])
    out["subject_hist_gpa"] = out["subject_hist_gpa"].fillna(overall_mean)

    out["instructor_hist_gpa"] = out["instructor_hist_gpa"].fillna(out["global_hist_gpa"])
    out["instructor_hist_gpa"] = out["instructor_hist_gpa"].fillna(overall_mean)

    return out


def time_based_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_terms = np.sort(df["time_index"].unique())
    split_idx = int(np.floor(len(unique_terms) * (1.0 - test_fraction)))
    split_idx = min(max(split_idx, 1), len(unique_terms) - 1)

    train_terms = set(unique_terms[:split_idx])
    test_terms = set(unique_terms[split_idx:])

    train_df = df[df["time_index"].isin(train_terms)].copy()
    test_df = df[df["time_index"].isin(test_terms)].copy()
    return train_df, test_df


def get_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "course_level",
        "class_size",
        "subject_hist_gpa",
        "instructor_hist_gpa",
    ]
    categorical_features = ["Term"]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def get_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def run_training_and_comparison(
    data_url: str,
    artifacts_dir: Path,
    test_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_url)
    engineered = build_features(raw)

    train_df, test_df = time_based_split(engineered, test_fraction=test_fraction)

    feature_cols = [
        "course_level",
        "class_size",
        "subject_hist_gpa",
        "instructor_hist_gpa",
        "Term",
    ]

    x_train = train_df[feature_cols]
    y_train = train_df["avg_gpa"]
    x_test = test_df[feature_cols]
    y_test = test_df["avg_gpa"]

    models = get_models(random_state=random_state)
    if len(models) < 3:
        print(
            "Warning: xgboost and/or lightgbm are not installed. "
            "Comparison will include available models only."
        )

    rows = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            preds = pipeline.predict(x_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        rows.append(
            {
                "model": model_name,
                "rmse": rmse,
                "r2": r2,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_terms": train_df["time_index"].nunique(),
                "test_terms": test_df["time_index"].nunique(),
            }
        )
        trained_pipelines[model_name] = pipeline

    results = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    results_path = artifacts_dir / "model_comparison.csv"
    results.to_csv(results_path, index=False)

    best_model_name = results.iloc[0]["model"]
    best_model_path = artifacts_dir / "best_model.joblib"
    dump(trained_pipelines[best_model_name], best_model_path)

    print("\nModel Comparison (lower RMSE is better):")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved comparison table to: {results_path}")
    print(f"Saved best model ({best_model_name}) to: {best_model_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare GPA prediction models on the UIUC GPA dataset."
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default="https://waf.cs.illinois.edu/discovery/gpa.csv",
        help="CSV URL or local file path for the UIUC GPA dataset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to store model outputs.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of most recent terms used for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training_and_comparison(
        data_url=args.data_url,
        artifacts_dir=Path(args.artifacts_dir),
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
