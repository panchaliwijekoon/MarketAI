import pandas as pd


file_path = (
    "DATASETS/furniture_tariffs/"
    "USA_SriLanka_940360_2022.xlsx"
)


# --------------------------------------------------
# CHECK AVAILABLE SHEETS
# --------------------------------------------------

excel_file = pd.ExcelFile(file_path)

print("\n==============================")
print("TARIFF FILE")
print("==============================")

print("\nSheet names:")
print(excel_file.sheet_names)


# --------------------------------------------------
# READ THE ACTUAL DATA SHEET
# --------------------------------------------------

tariff = pd.read_excel(
    file_path,
    sheet_name="Country-TariffData"
)


print("\n==============================")
print("COUNTRY-TARIFF DATA")
print("==============================")

print("\nColumns:")
print(tariff.columns.tolist())

print("\nShape:")
print(tariff.shape)

print("\nFirst 15 rows:")
print(
    tariff.head(15).to_string()
)