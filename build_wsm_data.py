import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATASETS"

PREDICTIONS_FILE = (
    DATA_DIR / "market_predictions_2026.csv"
)

GDP_FILE = (
    DATA_DIR / "worldbank_gdp_per_capita.csv.csv"
)

LPI_FILE = (
    DATA_DIR / "worldbank_logistics_lpi.csv.csv"
)

POLITICAL_FILE = (
    DATA_DIR / "worldbank_political_stability.csv.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "marketai_wsm_data.csv"
)


# --------------------------------------------------
# LOAD WORLD BANK DATA
# --------------------------------------------------

def load_worldbank(path):
    return pd.read_csv(
        path,
        skiprows=4
    )


print("\nLoading datasets...")


predictions = pd.read_csv(
    PREDICTIONS_FILE
)

gdp = load_worldbank(
    GDP_FILE
)

lpi = load_worldbank(
    LPI_FILE
)

political = load_worldbank(
    POLITICAL_FILE
)


# --------------------------------------------------
# PREPARE GDP
# Latest available year = 2025
# --------------------------------------------------

gdp_clean = gdp[
    [
        "Country Code",
        "Country Name",
        "2025"
    ]
].copy()


gdp_clean = gdp_clean.rename(
    columns={
        "Country Code": "country_code",
        "Country Name": "gdp_country_name",
        "2025": "gdp_per_capita"
    }
)


gdp_clean["gdp_per_capita"] = pd.to_numeric(
    gdp_clean["gdp_per_capita"],
    errors="coerce"
)


# --------------------------------------------------
# PREPARE LOGISTICS LPI
# Latest available year = 2022
# --------------------------------------------------

lpi_clean = lpi[
    [
        "Country Code",
        "Country Name",
        "2022"
    ]
].copy()


lpi_clean = lpi_clean.rename(
    columns={
        "Country Code": "country_code",
        "Country Name": "lpi_country_name",
        "2022": "logistics_lpi"
    }
)


lpi_clean["logistics_lpi"] = pd.to_numeric(
    lpi_clean["logistics_lpi"],
    errors="coerce"
)


# --------------------------------------------------
# PREPARE POLITICAL STABILITY
# Latest available year = 2024
# --------------------------------------------------

political_clean = political[
    [
        "Country Code",
        "Country Name",
        "2024"
    ]
].copy()


political_clean = political_clean.rename(
    columns={
        "Country Code": "country_code",
        "Country Name": "political_country_name",
        "2024": "political_stability"
    }
)


political_clean["political_stability"] = pd.to_numeric(
    political_clean["political_stability"],
    errors="coerce"
)


# The file appears to contain duplicate
# country rows, so combine them safely.

political_clean = (
    political_clean
    .groupby(
        "country_code",
        as_index=False
    )
    .agg(
        political_stability=(
            "political_stability",
            "mean"
        )
    )
)


# --------------------------------------------------
# PREPARE AI PREDICTIONS
# --------------------------------------------------

market_data = predictions.rename(
    columns={
        "reporterISO": "country_code",
        "reporterDesc": "country",
        "predicted_2026_import_value":
            "predicted_demand_2026",
        "predicted_growth_percent":
            "predicted_growth_2026"
    }
)


market_data = market_data[
    [
        "country_code",
        "country",
        "import_value",
        "growth_percent",
        "predicted_demand_2026",
        "predicted_growth_2026"
    ]
]


# --------------------------------------------------
# MERGE GDP
# --------------------------------------------------

market_data = market_data.merge(
    gdp_clean[
        [
            "country_code",
            "gdp_per_capita"
        ]
    ],
    on="country_code",
    how="left"
)


# --------------------------------------------------
# MERGE LOGISTICS
# --------------------------------------------------

market_data = market_data.merge(
    lpi_clean[
        [
            "country_code",
            "logistics_lpi"
        ]
    ],
    on="country_code",
    how="left"
)


# --------------------------------------------------
# MERGE POLITICAL STABILITY
# --------------------------------------------------

market_data = market_data.merge(
    political_clean,
    on="country_code",
    how="left"
)


# --------------------------------------------------
# DATA COVERAGE
# --------------------------------------------------

criteria_columns = [
    "predicted_demand_2026",
    "predicted_growth_2026",
    "gdp_per_capita",
    "logistics_lpi",
    "political_stability"
]


market_data["available_indicators"] = (
    market_data[
        criteria_columns
    ]
    .notna()
    .sum(axis=1)
)


market_data["data_coverage_percent"] = (
    market_data["available_indicators"]
    / len(criteria_columns)
) * 100


# --------------------------------------------------
# SORT BY AI-PREDICTED DEMAND
# --------------------------------------------------

market_data = market_data.sort_values(
    "predicted_demand_2026",
    ascending=False
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

market_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("WSM DATASET CREATED")
print("==============================")

print(
    "\nMarkets:",
    len(market_data)
)


print("\nMissing GDP:")
print(
    market_data["gdp_per_capita"]
    .isna()
    .sum()
)


print("\nMissing LPI:")
print(
    market_data["logistics_lpi"]
    .isna()
    .sum()
)


print("\nMissing Political Stability:")
print(
    market_data["political_stability"]
    .isna()
    .sum()
)


print("\nSample:")
print(
    market_data.head(15).to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)