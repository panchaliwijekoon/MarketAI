import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from product_config import PRODUCTS


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATASETS"
MODEL_DIR = BASE_DIR / "MODELS"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# WORLD BANK
# --------------------------------------------------

def load_worldbank(filename):

    return pd.read_csv(
        DATA_DIR / filename,
        skiprows=4
    )


gdp = load_worldbank(
    "worldbank_gdp_per_capita.csv.csv"
)

lpi = load_worldbank(
    "worldbank_logistics_lpi.csv.csv"
)

political = load_worldbank(
    "worldbank_political_stability.csv.csv"
)


gdp = gdp[
    ["Country Code", "2025"]
].rename(
    columns={
        "Country Code": "country_code",
        "2025": "gdp_per_capita"
    }
)


lpi = lpi[
    ["Country Code", "2022"]
].rename(
    columns={
        "Country Code": "country_code",
        "2022": "logistics_lpi"
    }
)


political = political[
    ["Country Code", "2024"]
].rename(
    columns={
        "Country Code": "country_code",
        "2024": "political_stability"
    }
)


political = (
    political
    .groupby("country_code", as_index=False)
    .agg(
        political_stability=(
            "political_stability",
            "mean"
        )
    )
)


# --------------------------------------------------
# TARIFF COUNTRY MAPPING
# --------------------------------------------------

COUNTRY_CODES = {

    "United States": "USA",
    "USA": "USA",

    "United Kingdom": "GBR",

    "Australia": "AUS",

    "Canada": "CAN",

    "India": "IND",

    "Japan": "JPN",

    "Maldives": "MDV",

    "Singapore": "SGP",

    "United Arab Emirates": "ARE",

    "European Union": "EU"
}


# --------------------------------------------------
# LOAD TARIFF FOLDER
# --------------------------------------------------

def load_tariffs(folder_name):

    if not folder_name:
        return None

    folder = DATA_DIR / folder_name

    if not folder.exists():
        return None


    records = []


    for file in folder.glob("*.xlsx"):

        try:

            tariff = pd.read_excel(
                file,
                sheet_name="Country-TariffData"
            )

        except Exception:

            continue


        for _, row in tariff.iterrows():

            reporter = str(
                row["Reporter"]
            ).strip()


            code = COUNTRY_CODES.get(
                reporter
            )


            if code and code != "EU":

                records.append(
                    {
                        "country_code": code,

                        "applied_tariff":
                            pd.to_numeric(
                                row["AppliedTariff"],
                                errors="coerce"
                            )
                    }
                )


    if not records:
        return None


    return (
        pd.DataFrame(records)
        .drop_duplicates(
            subset=["country_code"]
        )
    )


# --------------------------------------------------
# PROCESS EACH PRODUCT
# --------------------------------------------------

for hs_code, product in PRODUCTS.items():

    print("\n")
    print("=" * 50)
    print(
        "PROCESSING:",
        product["display_name"]
    )
    print("=" * 50)


    input_file = (
        DATA_DIR
        / product["dataset"]
    )


    # -----------------------------------
    # LOAD COMTRADE
    # -----------------------------------

    df = pd.read_csv(
        input_file,
        encoding="cp1252",
        usecols=range(47),
        low_memory=False
    )


    df["cmdCode"] = (
        df["cmdCode"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False
        )
        .str.strip()
    )


    df["refYear"] = pd.to_numeric(
        df["refYear"],
        errors="coerce"
    )


    df["primaryValue"] = pd.to_numeric(
        df["primaryValue"],
        errors="coerce"
    )


    # -----------------------------------
    # FILTER PRODUCT
    # -----------------------------------

    product_data = df[
        (df["cmdCode"] == hs_code)
        &
        (df["flowDesc"] == "Import")
        &
        (df["partnerDesc"] == "World")
    ].copy()


    product_data = product_data.dropna(
        subset=[
            "reporterISO",
            "reporterDesc",
            "refYear",
            "primaryValue"
        ]
    )


    print(
        "Raw matching records:",
        len(product_data)
    )


    # -----------------------------------
    # COUNTRY/YEAR DATA
    # -----------------------------------

    yearly = (
        product_data
        .groupby(
            [
                "reporterISO",
                "reporterDesc",
                "refYear"
            ],
            as_index=False
        )
        .agg(
            import_value=(
                "primaryValue",
                "max"
            )
        )
    )


    yearly = yearly.sort_values(
        [
            "reporterISO",
            "refYear"
        ]
    )


    # -----------------------------------
    # PREVIOUS YEAR
    # -----------------------------------

    yearly["previous_year"] = (
        yearly
        .groupby("reporterISO")[
            "refYear"
        ]
        .shift(1)
    )


    yearly[
        "previous_import_value"
    ] = (
        yearly
        .groupby("reporterISO")[
            "import_value"
        ]
        .shift(1)
    )


    wrong_previous = (
        yearly["refYear"]
        -
        yearly["previous_year"]
        != 1
    )


    yearly.loc[
        wrong_previous,
        "previous_import_value"] = np.nan


    # -----------------------------------
    # GROWTH
    # -----------------------------------

    yearly[ "growth_percent" ] = (
        (
            yearly["import_value"]
            -
            yearly[
                "previous_import_value"
            ]
        )
        /
        yearly[
            "previous_import_value"
        ]
    ) * 100


    yearly["growth_percent"
 ] = yearly[
        "growth_percent"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )


    # -----------------------------------
    # NEXT-YEAR TARGET
    # -----------------------------------

    yearly["next_year"] = (
        yearly
        .groupby("reporterISO")[
            "refYear"
        ]
        .shift(-1)
    )


    yearly[
        "next_year_import_value"
    ] = (
        yearly
        .groupby("reporterISO")[
            "import_value"
        ]
        .shift(-1)
    )


    wrong_next = (
        yearly["next_year"]
        -
        yearly["refYear"]
        != 1
    )


    yearly.loc[
        wrong_next,
        "next_year_import_value"
    ] = np.nan


    # -----------------------------------
    # TRAINING DATA
    # -----------------------------------

    training = yearly.dropna(
        subset=[
            "previous_import_value",
            "import_value",
            "growth_percent",
            "next_year_import_value"
        ]
    ).copy()


    print(
        "Training rows:",
        len(training)
    )


    if len(training) < 20:

        print(
            "Not enough training data - skipped."
        )

        continue


    # -----------------------------------
    # FEATURES
    # -----------------------------------

    training["log_previous"] = (
        np.log1p(
            training[
                "previous_import_value"
            ]
        )
    )


    training["log_current"] = (
        np.log1p(
            training["import_value"]
        )
    )


    training["log_target"] = (
        np.log1p(
            training[
                "next_year_import_value"
            ]
        )
    )


    features = [
        "log_previous",
        "log_current",
        "growth_percent"
    ]


    X = training[
        features
    ]

    y = training[
        "log_target"
    ]


    # -----------------------------------
    # QUICK HISTORICAL EVALUATION
    # -----------------------------------

    latest_training_year = int(
        training[
            "refYear"
        ].max()
    )


    train_mask = (
        training["refYear"]
        <
        latest_training_year
    )


    test_mask = (
        training["refYear"]
        ==
        latest_training_year
    )


    if test_mask.sum() > 5:

        test_model = (
            LinearRegression()
        )


        test_model.fit(
            X[train_mask],
            y[train_mask]
        )


        test_prediction = (
            test_model.predict(
                X[test_mask]
            )
        )


        actual = np.expm1(
            y[test_mask]
        )


        predicted = np.expm1(
            test_prediction
        )


        score = r2_score(
            actual,
            predicted
        )


        print(
            "Held-out R²:",
            round(score, 4)
        )


    # -----------------------------------
    # FINAL MODEL
    # -----------------------------------

    model = LinearRegression()

    model.fit(X,y)


    model_file = ( MODEL_DIR / f"demand_model_{hs_code}.pkl" )


    joblib.dump( model, model_file )


    # -----------------------------------
    # LATEST YEAR
    # -----------------------------------

    latest_year = int(
        yearly["refYear"].max())


    latest = yearly[yearly["refYear"] == latest_year].copy()


    latest = latest.dropna(
        subset=[
            "previous_import_value",
            "growth_percent",
            "import_value"
        ]
    )


    latest["log_previous"] = (np.log1p( latest[ "previous_import_value"] ) )


    latest["log_current"] = (np.log1p(latest["import_value"]))


    # -----------------------------------
    # PREDICT NEXT YEAR
    # -----------------------------------

    prediction_log = ( model.predict(latest[features]))


    predictions = np.expm1( prediction_log)


    predictions = np.maximum(predictions, 0 )


    latest[ "predicted_demand_2026" ] = predictions


    latest[ "predicted_growth_2026"  ] = ((

            latest["predicted_demand_2026"]
            -
            latest["import_value"]
        )
        /
        latest["import_value"]
    ) * 100


    # -----------------------------------
    # CLEAN OUTPUT
    # -----------------------------------

    output = latest[
        [
            "reporterISO",
            "reporterDesc",
            "import_value",
            "growth_percent",
            "predicted_demand_2026",
            "predicted_growth_2026"
        ]
    ].rename(
        columns={
            "reporterISO":
                "country_code",

            "reporterDesc":
                "country"
        }
    )


    # -----------------------------------
    # WORLD BANK MERGES
    # -----------------------------------

    output = output.merge(gdp,
        on="country_code",
        how="left" )


    output = output.merge(
        lpi,
        on="country_code",
        how="left"
    )


    output = output.merge(
        political,
        on="country_code",
        how="left"
    )


    # -----------------------------------
    # TARIFF
    # -----------------------------------

    tariffs = load_tariffs( product.get( "tariff_folder" ) )


    if tariffs is not None:

        output = output.merge(
            tariffs,
            on="country_code",
            how="inner"
        )

        print( "Tariff data included.")

    else:

        output[ "applied_tariff" ] = np.nan

        print("No tariff dataset - " "5-criterion ranking will be used." )


    # -----------------------------------
    # REMOVE MISSING CORE DATA
    # -----------------------------------

    output = output.dropna(
        subset=[
            "predicted_demand_2026",
            "predicted_growth_2026",
            "gdp_per_capita",
            "logistics_lpi",
            "political_stability"
        ]
    )


    # -----------------------------------
    # SAVE
    # -----------------------------------

    output_file = (DATA_DIR  / f"marketai_final_{hs_code}.csv" )


    output.to_csv( output_file, index=False )


    print( "Markets available:", len(output))

    print( "Created:",output_file.name)


print("\n==============================")
print("MULTI-PRODUCT BUILD COMPLETE")
print("==============================")