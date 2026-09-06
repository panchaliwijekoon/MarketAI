import pandas as pd


def load_worldbank(path):
    return pd.read_csv(
        path,
        skiprows=4
    )


gdp = load_worldbank(
    "DATASETS/worldbank_gdp_per_capita.csv.csv"
)

lpi = load_worldbank(
    "DATASETS/worldbank_logistics_lpi.csv.csv"
)

political = load_worldbank(
    "DATASETS/worldbank_political_stability.csv.csv"
)


def inspect_dataset(name, df):

    print("\n==============================")
    print(name)
    print("==============================")

    print("\nIndicators:")

    print(
        df[
            [
                "Indicator Name",
                "Indicator Code"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    year_columns = [
        column
        for column in df.columns
        if column.isdigit()
    ]

    print("\nLatest years containing data:")

    for year in reversed(year_columns):

        values = pd.to_numeric(
            df[year],
            errors="coerce"
        )

        count = values.notna().sum()

        if count > 0:
            print(
                year,
                "-",
                count,
                "values"
            )

        # Only show recent years
        if year == "2020":
            break


inspect_dataset(
    "GDP PER CAPITA",
    gdp
)

inspect_dataset(
    "LOGISTICS LPI",
    lpi
)

inspect_dataset(
    "POLITICAL STABILITY",
    political
)