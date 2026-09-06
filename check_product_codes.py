import pandas as pd
from pathlib import Path


DATA_DIR = Path("DATASETS")


files = {
    "Furniture": "comtrade_furniture.csv",
    "Tea": "comtrade_tea.csv",
    "Apparel": "comtrade_apparel.csv",
    "Jewellery": "comtrade_jewellery.csv"
}


for product_name, filename in files.items():

    path = DATA_DIR / filename

    print("\n==============================")
    print(product_name.upper())
    print("==============================")

    try:

        df = pd.read_csv(
            path,
            encoding="cp1252",
            usecols=range(47),
            low_memory=False
        )

    except Exception:

        df = pd.read_csv(
            path,
            encoding="cp1252",
            low_memory=False
        )


    codes = (
        df[
            [
                "cmdCode",
                "cmdDesc"
            ]
        ]
        .drop_duplicates()
    )


    print(
        codes.to_string(
            index=False
        )
    )