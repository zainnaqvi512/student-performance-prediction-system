"""Training and inference for final-score prediction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


FEATURES = ["attendance", "quiz_score", "assignment_score", "midterm_score", "previous_score"]
TARGET = "final_score"
LABELS = ["At Risk", "Average", "Excellent"]
MIN_ROWS = 30


def risk_level(score: float) -> str:
    if score < 50:
        return "At Risk"
    if score < 80:
        return "Average"
    return "Excellent"


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame = frame[FEATURES + [TARGET]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[TARGET].isna().any():
        raise ValueError("Every row needs a numeric final_score.")
    if frame[FEATURES].isna().all(axis=1).any():
        raise ValueError("Each row needs at least one assessment or attendance value.")
    for column in frame.columns:
        invalid = frame[column].notna() & ~frame[column].between(0, 100)
        if invalid.any():
            raise ValueError(f"{column} must be between 0 and 100 (or blank for input features).")
    if len(frame) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} historical rows are needed; found {len(frame)}.")
    if frame[TARGET].nunique() < 2:
        raise ValueError("The final_score column needs at least two distinct values.")
    return frame


def _pipeline(estimator):
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), FEATURES)
        ], remainder="drop")),
        ("model", estimator),
    ])


def train_models(frame: pd.DataFrame):
    """Return the best fitted model and honest held-out comparison metrics."""
    frame = validate_training_frame(frame)
    x_development, x_test, y_development, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.2, random_state=42
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development, y_development, test_size=0.25, random_state=42
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1),
    }
    metrics = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(x_train, y_train)
        validation_prediction = np.clip(pipeline.predict(x_validation), 0, 100)
        validation_mae = float(mean_absolute_error(y_validation, validation_prediction))
        pipeline.fit(x_development, y_development)
        predicted = np.clip(pipeline.predict(x_test), 0, 100)
        actual_labels = [risk_level(score) for score in y_test]
        predicted_labels = [risk_level(score) for score in predicted]
        metrics.append({
            "name": name,
            "validation_mae": round(validation_mae, 2),
            "mae": round(float(mean_absolute_error(y_test, predicted)), 2),
            "rmse": round(float(math.sqrt(mean_squared_error(y_test, predicted))), 2),
            "r2": round(float(r2_score(y_test, predicted)), 3),
            "accuracy": round(float(accuracy_score(actual_labels, predicted_labels) * 100), 1),
            "confusion": confusion_matrix(actual_labels, predicted_labels, labels=LABELS).tolist(),
        })
        fitted[name] = pipeline
    winner = min(metrics, key=lambda row: row["validation_mae"])["name"]
    # Refit the selected architecture on all available historical data after evaluation.
    best = fitted[winner]
    best.fit(frame[FEATURES], frame[TARGET])
    return best, metrics, winner, len(x_test)


def predict_score(model, values: dict) -> float:
    row = pd.DataFrame([{key: float(values[key]) for key in FEATURES}])
    return round(float(np.clip(model.predict(row)[0], 0, 100)), 1)


def demo_frame(rows: int = 240) -> pd.DataFrame:
    """Deterministic illustration data. Never presented as real academic evidence."""
    rng = np.random.default_rng(23)
    attendance = rng.uniform(35, 100, rows)
    quiz = rng.uniform(20, 100, rows)
    assignment = rng.uniform(25, 100, rows)
    midterm = rng.uniform(15, 100, rows)
    previous = rng.uniform(25, 100, rows)
    score = (0.13 * attendance + 0.14 * quiz + 0.19 * assignment +
             0.38 * midterm + 0.16 * previous + rng.normal(0, 6, rows))
    return pd.DataFrame({
        "attendance": attendance.round(1),
        "quiz_score": quiz.round(1),
        "assignment_score": assignment.round(1),
        "midterm_score": midterm.round(1),
        "previous_score": previous.round(1),
        "final_score": np.clip(score, 0, 100).round(1),
    })
