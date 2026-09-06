import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATASETS"


MARKET_FILE = (
    DATA_DIR
    / "marketai_wsm_data.csv"
)

TARIFF_FILE = (
    DATA_DIR
    / "furniture_tariffs_clean.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "marketai_final_data.csv"
)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

markets = pd.read_csv(
    MARKET_FILE
)

tariffs = pd.read_csv(
    TARIFF_FILE
)


# --------------------------------------------------
# REMOVE EU AGGREGATE FOR NOW
# --------------------------------------------------

tariffs = tariffs[
    tariffs["country_code"] != "EU"
].copy()


# --------------------------------------------------
# KEEP ONLY NEEDED TARIFF FIELDS
# --------------------------------------------------

tariffs = tariffs[
    [
        "country_code",
        "applied_tariff",
        "mfn_rate",
        "year"
    ]
]


tariffs = tariffs.rename(
    columns={
        "year":
            "tariff_year"
    }
)


# --------------------------------------------------
# MERGE
# --------------------------------------------------

final_data = markets.merge(
    tariffs,
    on="country_code",
    how="inner"
)


# --------------------------------------------------
# REMOVE RECORDS MISSING CORE WSM DATA
# --------------------------------------------------

required_columns = [

    "predicted_demand_2026",

    "predicted_growth_2026",

    "gdp_per_capita",

    "logistics_lpi",

    "political_stability",

    "applied_tariff"
]


final_data = final_data.dropna(
    subset=required_columns
)


# --------------------------------------------------
# SORT
# --------------------------------------------------

final_data = final_data.sort_values(
    "predicted_demand_2026",
    ascending=False
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

final_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n==============================")
print("FINAL MARKET DATA")
print("==============================")

print(
    "\nMarkets available for WSM:",
    len(final_data)
)


print(
    "\nCountries:"
)

print(
    final_data[
        [
            "country_code",
            "country",
            "predicted_demand_2026",
            "predicted_growth_2026",
            "gdp_per_capita",
            "logistics_lpi",
            "political_stability",
            "applied_tariff"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)