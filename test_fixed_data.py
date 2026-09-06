import pandas as pd


file_path = "DATASETS/comtrade_furniture.csv"


# Read only the 47 real columns
df = pd.read_csv(
    file_path,
    encoding="cp1252",
    usecols=range(47),
    low_memory=False
)


print("\n==============================")
print("DATA LOADED")
print("==============================")

print("Shape:")
print(df.shape)


print("\nFIRST ROW:")
print(
    df[
        [
            "typeCode",
            "freqCode",
            "refPeriodId",
            "refYear",
            "reporterISO",
            "reporterDesc",
            "flowCode",
            "flowDesc",
            "partnerISO",
            "partnerDesc",
            "cmdCode",
            "cmdDesc",
            "primaryValue"
        ]
    ].head(1)
)


print("\nYEARS FOUND:")
print(
    sorted(
        df["refYear"]
        .dropna()
        .unique()
    )
)


print("\nTRADE FLOWS:")
print(
    df["flowDesc"]
    .value_counts()
)


print("\nPARTNERS:")
print(
    df["partnerDesc"]
    .value_counts()
    .head(10)
)


print("\nPRIMARY VALUE:")
print(
    df["primaryValue"]
    .describe()
)