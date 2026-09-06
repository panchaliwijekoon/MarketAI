import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "DATASETS" / "comtrade_furniture.csv"

YEARLY_OUTPUT = (
    BASE_DIR / "DATASETS" / "furniture_yearly_market_data.csv"
)

TRAINING_OUTPUT = (
    BASE_DIR / "DATASETS" / "market_training_data.csv"
)

LATEST_OUTPUT = (
    BASE_DIR / "DATASETS" / "market_latest_data.csv"
)


# --------------------------------------------------
# 1. LOAD RAW COMTRADE DATA
# --------------------------------------------------

print("\nLoading UN Comtrade furniture data...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="cp1252",

    # CSV contains one unwanted trailing field
    usecols=range(47),

    low_memory=False
)

print("Raw dataset shape:", df.shape)


# --------------------------------------------------
# 2. CLEAN IMPORTANT COLUMNS
# --------------------------------------------------

df["refYear"] = pd.to_numeric(
    df["refYear"],
    errors="coerce"
)

df["primaryValue"] = pd.to_numeric(
    df["primaryValue"],
    errors="coerce"
)

# Make HS code consistent
df["cmdCode"] = (
    df["cmdCode"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)


# --------------------------------------------------
# 3. FILTER FURNITURE MARKET DATA
# --------------------------------------------------

# We only want:
# HS 940360
# Import flow
# Partner = World

furniture = df[
    (df["cmdCode"] == "940360")
    & (df["flowDesc"] == "Import")
    & (df["partnerDesc"] == "World")
].copy()


print("\nFurniture import records:", len(furniture))


# Remove invalid observations
furniture = furniture.dropna(
    subset=[
        "reporterISO",
        "reporterDesc",
        "refYear",
        "primaryValue"
    ]
)

furniture = furniture[
    furniture["primaryValue"] >= 0
]


# --------------------------------------------------
# 4. REMOVE EXACT DUPLICATE VALUES
# --------------------------------------------------

furniture = furniture.drop_duplicates(
    subset=[
        "reporterISO",
        "reporterDesc",
        "refYear",
        "primaryValue"
    ]
)


# --------------------------------------------------
# 5. CHECK COUNTRY-YEAR DUPLICATES
# --------------------------------------------------

duplicate_check = (
    furniture
    .groupby(
        [
            "reporterISO",
            "reporterDesc",
            "refYear"
        ]
    )
    .size()
    .reset_index(name="records")
)

duplicate_check = duplicate_check[
    duplicate_check["records"] > 1
]


if len(duplicate_check) > 0:

    print(
        "\nWarning: Some country-year combinations "
        "contain multiple records."
    )

    print(duplicate_check.head(10))


# --------------------------------------------------
# 6. CREATE ONE MARKET VALUE PER COUNTRY/YEAR
# --------------------------------------------------

# If duplicate totals remain, take the largest
# reported total for that country-year.

yearly = (
    furniture
    .groupby(
        [
            "reporterISO",
            "reporterDesc",
            "refYear"
        ],
        as_index=False
    )
    .agg(
        import_value=("primaryValue", "max")
    )
)


yearly = yearly.sort_values(
    [
        "reporterISO",
        "refYear"
    ]
)


print(
    "\nCountry-year observations:",
    len(yearly)
)


# --------------------------------------------------
# 7. CREATE PREVIOUS-YEAR VALUE
# --------------------------------------------------

yearly["previous_year"] = (
    yearly
    .groupby("reporterISO")["refYear"]
    .shift(1)
)


yearly["previous_import_value"] = (
    yearly
    .groupby("reporterISO")["import_value"]
    .shift(1)
)


# Only accept consecutive years
not_consecutive_previous = (
    yearly["refYear"]
    - yearly["previous_year"]
    != 1
)

yearly.loc[
    not_consecutive_previous,
    "previous_import_value"
] = np.nan


# --------------------------------------------------
# 8. CALCULATE MARKET GROWTH
# --------------------------------------------------

yearly["growth_percent"] = (
    (
        yearly["import_value"]
        - yearly["previous_import_value"]
    )
    / yearly["previous_import_value"]
) * 100


# Remove infinite growth values
yearly["growth_percent"] = (
    yearly["growth_percent"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# --------------------------------------------------
# 9. CREATE NEXT-YEAR AI TARGET
# --------------------------------------------------

yearly["next_year"] = (
    yearly
    .groupby("reporterISO")["refYear"]
    .shift(-1)
)


yearly["next_year_import_value"] = (
    yearly
    .groupby("reporterISO")["import_value"]
    .shift(-1)
)


# Make sure the target really is the next year
not_consecutive_next = (
    yearly["next_year"]
    - yearly["refYear"]
    != 1
)

yearly.loc[
    not_consecutive_next,
    "next_year_import_value"
] = np.nan


# --------------------------------------------------
# 10. CREATE MACHINE-LEARNING TRAINING DATA
# --------------------------------------------------

training_data = yearly.dropna(
    subset=[
        "import_value",
        "previous_import_value",
        "growth_percent",
        "next_year_import_value"
    ]
).copy()


training_data = training_data[
    [
        "reporterISO",
        "reporterDesc",
        "refYear",
        "previous_import_value",
        "import_value",
        "growth_percent",
        "next_year_import_value"
    ]
]


# --------------------------------------------------
# 11. GET LATEST MARKET DATA FOR FUTURE PREDICTION
# --------------------------------------------------

latest_year = int(
    yearly["refYear"].max()
)

latest_data = yearly[
    yearly["refYear"] == latest_year
].copy()


latest_data = latest_data.dropna(
    subset=[
        "previous_import_value",
        "growth_percent",
        "import_value"
    ]
)


latest_data = latest_data[
    [
        "reporterISO",
        "reporterDesc",
        "refYear",
        "previous_import_value",
        "import_value",
        "growth_percent"
    ]
]


# --------------------------------------------------
# 12. SAVE CLEAN DATASETS
# --------------------------------------------------

yearly.to_csv(
    YEARLY_OUTPUT,
    index=False
)

training_data.to_csv(
    TRAINING_OUTPUT,
    index=False
)

latest_data.to_csv(
    LATEST_OUTPUT,
    index=False
)


# --------------------------------------------------
# 13. DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("DATA PREPARATION COMPLETE")
print("==============================")

print(
    "\nYears available:",
    sorted(
        yearly["refYear"].unique()
    )
)

print(
    "\nCountries available:",
    yearly["reporterISO"].nunique()
)

print(
    "\nTraining observations:",
    len(training_data)
)

print(
    "\nLatest prediction year:",
    latest_year
)

print(
    "\nMarkets available for future prediction:",
    len(latest_data)
)


print("\nTRAINING DATA SAMPLE:")

print(
    training_data.head(10)
)


print("\nLATEST MARKET SAMPLE:")

print(
    latest_data.head(10)
)


print(
    "\nFiles created:"
)

print(YEARLY_OUTPUT.name)
print(TRAINING_OUTPUT.name)
print(LATEST_OUTPUT.name)