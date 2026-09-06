import pandas as pd


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def load_worldbank_file(file_path):

    return pd.read_csv(
        file_path,
        skiprows=4,
        encoding="utf-8"
    )


# --------------------------------------------------
# LOAD GDP
# --------------------------------------------------

gdp = load_worldbank_file(
    "DATASETS/worldbank_gdp_per_capita.csv.csv"
)


# --------------------------------------------------
# LOAD LOGISTICS LPI
# --------------------------------------------------

lpi = load_worldbank_file(
    "DATASETS/worldbank_logistics_lpi.csv.csv"
)


# --------------------------------------------------
# LOAD POLITICAL STABILITY
# --------------------------------------------------

risk = load_worldbank_file(
    "DATASETS/worldbank_political_stability.csv.csv"
)


# --------------------------------------------------
# GDP CHECK
# --------------------------------------------------

print("\n==============================")
print("GDP PER CAPITA")
print("==============================")

print("\nColumns:")
print(gdp.columns.tolist())

print("\nFirst 3 rows:")
print(gdp.head(3))

print("\nShape:")
print(gdp.shape)


# --------------------------------------------------
# LOGISTICS CHECK
# --------------------------------------------------

print("\n==============================")
print("LOGISTICS LPI")
print("==============================")

print("\nColumns:")
print(lpi.columns.tolist())

print("\nFirst 3 rows:")
print(lpi.head(3))

print("\nShape:")
print(lpi.shape)


# --------------------------------------------------
# POLITICAL STABILITY CHECK
# --------------------------------------------------

print("\n==============================")
print("POLITICAL STABILITY")
print("==============================")

print("\nColumns:")
print(risk.columns.tolist())

print("\nFirst 3 rows:")
print(risk.head(3))

print("\nShape:")
print(risk.shape)