import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAINING_FILE = (
    BASE_DIR / "DATASETS" / "market_training_data.csv"
)

LATEST_FILE = (
    BASE_DIR / "DATASETS" / "market_latest_data.csv"
)

MODEL_FILE = (
    BASE_DIR / "market_demand_model.pkl"
)

PREDICTION_FILE = (
    BASE_DIR / "DATASETS" / "market_predictions_2026.csv"
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading training data...")

training = pd.read_csv(TRAINING_FILE)

latest = pd.read_csv(LATEST_FILE)

print("Training observations:", len(training))
print("Latest markets:", len(latest))


# --------------------------------------------------
# 2. CREATE MACHINE-LEARNING FEATURES
# --------------------------------------------------

# Trade values are very large and highly skewed.
# log1p reduces the effect of extremely large markets.

training["log_previous_import"] = np.log1p(
    training["previous_import_value"]
)

training["log_current_import"] = np.log1p(
    training["import_value"]
)

training["log_target"] = np.log1p(
    training["next_year_import_value"]
)


# These are the variables the AI uses as input
features = [
    "log_previous_import",
    "log_current_import",
    "growth_percent"
]


# X = input variables
X = training[features]

# y = value the AI must learn to predict
y = training["log_target"]


# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

# Use the latest historical year as the test period.
# This is more realistic than randomly mixing years.

evaluation_year = int(
    training["refYear"].max()
)


train_mask = (
    training["refYear"] < evaluation_year
)

test_mask = (
    training["refYear"] == evaluation_year
)


X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("\n==============================")
print("TRAIN / TEST SPLIT")
print("==============================")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Evaluation current year:",
    evaluation_year
)

print(
    "Evaluation target year:",
    evaluation_year + 1
)


# --------------------------------------------------
# 4. CREATE THE AI MODEL
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 6. MAKE TEST PREDICTIONS
# --------------------------------------------------

predicted_log = model.predict(
    X_test
)


# Convert predictions back from logarithmic values
predicted_values = np.expm1(predicted_log)

actual_values = np.expm1(y_test)

# --------------------------------------------------
# SAVE MODEL EVALUATION DATA
# --------------------------------------------------

evaluation_results = training.loc[
    test_mask,
    [
        "reporterISO",
        "reporterDesc",
        "refYear",
        "next_year_import_value"
    ]
].copy()


evaluation_results["predicted_next_year_value" ] = predicted_values


evaluation_results = evaluation_results.rename(
    columns={
        "next_year_import_value":
            "actual_2025_value",

        "predicted_next_year_value":
            "predicted_2025_value"
    }
)


evaluation_results = evaluation_results.sort_values(
    "actual_2025_value",
    ascending=False
)


evaluation_results.to_csv(
    "DATASETS/model_evaluation_2025.csv",
    index=False
)


print(
    "\nCreated:",
    "model_evaluation_2025.csv")

# Prediction cannot logically be negative
predicted_values = np.maximum(
    predicted_values,
    0
)


# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(
    actual_values,
    predicted_values
)

rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        predicted_values
    )
)

r2 = r2_score(
    actual_values,
    predicted_values
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    "Mean Absolute Error:",
    round(mae, 2)
)

print(
    "Root Mean Squared Error:",
    round(rmse, 2)
)

print(
    "R² Score:",
    round(r2, 4)
)


# --------------------------------------------------
# 8. SHOW MODEL COEFFICIENTS
# --------------------------------------------------

print("\n==============================")
print("MODEL COEFFICIENTS")
print("==============================")

print(
    "Intercept:",
    model.intercept_
)


for name, coefficient in zip(
    features,
    model.coef_
):

    print(
        name,
        "=",
        round(coefficient, 6)
    )


# --------------------------------------------------
# 9. TRAIN FINAL MODEL USING ALL HISTORICAL DATA
# --------------------------------------------------

print("\nTraining final model using all data...")

final_model = LinearRegression()

final_model.fit(
    X,
    y
)


# --------------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    final_model,
    MODEL_FILE
)

print(
    "\nModel saved as:",
    MODEL_FILE.name
)


# --------------------------------------------------
# 11. PREPARE 2025 DATA FOR 2026 PREDICTION
# --------------------------------------------------

latest["log_previous_import"] = np.log1p(
    latest["previous_import_value"]
)

latest["log_current_import"] = np.log1p(
    latest["import_value"]
)


X_future = latest[features]


# --------------------------------------------------
# 12. PREDICT 2026 DEMAND
# --------------------------------------------------

future_log_predictions = (
    final_model.predict(X_future)
)

future_predictions = np.expm1(
    future_log_predictions
)

future_predictions = np.maximum(
    future_predictions,
    0
)


latest["predicted_2026_import_value"] = (
    future_predictions
)


# --------------------------------------------------
# 13. CALCULATE PREDICTED GROWTH
# --------------------------------------------------

latest["predicted_growth_percent"] = (
    (
        latest["predicted_2026_import_value"]
        - latest["import_value"]
    )
    / latest["import_value"]
) * 100


# --------------------------------------------------
# 14. SORT MARKETS
# --------------------------------------------------

predictions = latest[
    [
        "reporterISO",
        "reporterDesc",
        "refYear",
        "import_value",
        "growth_percent",
        "predicted_2026_import_value",
        "predicted_growth_percent"
    ]
].copy()


predictions = predictions.sort_values(
    "predicted_2026_import_value",
    ascending=False
)


# --------------------------------------------------
# 15. SAVE PREDICTIONS
# --------------------------------------------------

predictions.to_csv(
    PREDICTION_FILE,
    index=False
)


print("\n==============================")
print("2026 AI MARKET PREDICTIONS")
print("==============================")

print(
    predictions.head(15).to_string(
        index=False
    )
)


print(
    "\nPrediction file created:",
    PREDICTION_FILE.name
)

print("\nAI TRAINING COMPLETE.")

# --------------------------------------------------
# BASELINE COMPARISON
# --------------------------------------------------

# Simple baseline:
# assume next year's demand equals current demand

baseline_values = training.loc[
    test_mask,
    "import_value"
].values


baseline_mae = mean_absolute_error(
    actual_values,
    baseline_values
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        baseline_values
    )
)

baseline_r2 = r2_score(
    actual_values,
    baseline_values
)


print("\n==============================")
print("BASELINE COMPARISON")
print("==============================")

print("Linear Regression")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 4))


print("\nNaive Baseline")
print("MAE:", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R²:", round(baseline_r2, 4))