import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TARIFF_FOLDER = (
    BASE_DIR
    / "DATASETS"
    / "furniture_tariffs"
)

OUTPUT_FILE = (
    BASE_DIR
    / "DATASETS"
    / "furniture_tariffs_clean.csv"
)


# --------------------------------------------------
# COUNTRY NAME → ISO CODE
# --------------------------------------------------

country_codes = {

    "United States": "USA",

    "United Kingdom": "GBR",

    "United Arab Emirates": "ARE",

    "Australia": "AUS",

    "Canada": "CAN",

    "India": "IND",

    "Japan": "JPN",

    "Maldives": "MDV",

    "Singapore": "SGP",

    # EU is not a normal individual country code.
    # We keep it separately for now.
    "European Union": "EU"
}


# --------------------------------------------------
# READ ALL EXCEL FILES
# --------------------------------------------------

records = []


for file_path in TARIFF_FOLDER.glob("*.xlsx"):

    print(
        "\nReading:",
        file_path.name
    )

    try:

        df = pd.read_excel(
            file_path,
            sheet_name="Country-TariffData"
        )

    except Exception as error:

        print(
            "Could not read:",
            error
        )

        continue


    if df.empty:

        print(
            "No tariff data found."
        )

        continue


    # There should normally be one row.
    for _, row in df.iterrows():

        reporter = str(
            row["Reporter"]
        ).strip()

        country_code = (
            country_codes.get(
                reporter
            )
        )


        records.append(
            {
                "country_code":
                    country_code,

                "country":
                    reporter,

                "year":
                    row["Year"],

                "partner":
                    row["Partner"],

                "product":
                    row["Product"],

                "mfn_rate":
                    row["MFNRate"],

                "applied_tariff":
                    row["AppliedTariff"],

                "tariff_lines":
                    row["TotalTariffLines"],

                "is_traded":
                    row["IsTraded"],

                "source_file":
                    file_path.name
            }
        )


# --------------------------------------------------
# CREATE CLEAN DATAFRAME
# --------------------------------------------------

tariffs = pd.DataFrame(
    records
)


# Convert tariff fields to numbers
tariffs["applied_tariff"] = (
    pd.to_numeric(
        tariffs["applied_tariff"],
        errors="coerce"
    )
)


tariffs["mfn_rate"] = (
    pd.to_numeric(
        tariffs["mfn_rate"],
        errors="coerce"
    )
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("CLEAN TARIFF DATA")
print("==============================")

print(
    tariffs[
        [
            "country_code",
            "country",
            "year",
            "mfn_rate",
            "applied_tariff",
            "is_traded"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

tariffs.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)