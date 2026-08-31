import pandas as pd
import numpy as np

pd.set_option("display.float_format","{:,.2f}".format)
#load comtrade furniture data 

file = "DATASETS/comtrade_furniture.csv"
df = pd.read_csv(file, encoding="cp1252", index_col=False)

print(df.head())
print()
print("Columns:")
print(df.columns.tolist())
print()
print("nUMBER OF ROWS:", len(df))

print("\nYears:")
print(sorted(df["refYear"].dropna().unique()))

print("\nTrade Flows:")
print(df["flowDesc"].value_counts())

print("\nProduct codes:")
print(df[["cmdCode", "cmdDesc"]].drop_duplicates().head(30))

print("\nReporter countries:")
print(df["reporterDesc"].value_counts().head(20))

print("\nPartner countries:")
print(df["partnerDesc"].value_counts().head(20))

#___________________________________________________________________

#creating the CLEANED dataset for furniture
Furniture = df[df["cmdCode"].astype(str)=="940360"].copy()
print("\nHS 940360 rows:", len(Furniture))

#keep total imports from the world
world_imports = Furniture[ Furniture["partnerDesc"]== "World"].copy()
print("World import rows:", len(world_imports))

print("\nSample:")
print(world_imports[["refYear","reporterISO","partnerDesc","primaryValue"]].head(20))


#checking for duplicates 
counts =(world_imports.groupby(["reporterISO", "refYear"]).size().reset_index(name="rows"))

print("\nMaximum rows for one country/year:")
print(counts["rows"].max())
print("\nDuplicates if any:")
print(counts[counts["rows"]>1])

#creating ONE ROW PER COUNTRY -organising the data
market_data =(world_imports[["reporterISO","reporterDesc","refYear","primaryValue"]].pivot(
    index=["reporterISO","reporterDesc"],
    columns="refYear",
    values="primaryValue"
).reset_index())

print("\nMarket data:")
print(market_data.head(20))

print("\nData coverade by year:")
for year in [2021,2022,2023,2024,2025]:
    available = market_data[year].notna().sum()
    print(year,":",available,"countries")

#CALCULATING THE RANKING_____________________________________________________________________________________
#main comparison year: 2024
latest_year = 2024

market_data["demand"] = market_data[latest_year]

top_demand = (market_data.dropna(subset=["demand"]).sort_values("demand",ascending=False))

print("\nTp[ 10 markets by 2024 import demand:")
print( top_demand[["reporterISO","reporterDesc","demand"]].head(10).to_string(index=False))

#calculating growth from 2021-2024_________________________________________________________________________
#CAGR= Compound Annual Growth Rate
valid_growth = (market_data[2021].notna()&
                market_data[2024].notna()&
                (market_data[2021]>0))

market_data.loc[valid_growth,"growth_cagr"] = ((market_data.loc[valid_growth,2024]
                                               /market_data.loc[valid_growth,2021])**(1/3)-1)

market_data["growth_percentage"]=(market_data["growth_cagr"]*100)

print("\nTop 10 fastest growing markets:")
print(market_data[["reporterISO","reporterDesc",2021,2024,"growth_percentage"]]
      .dropna(subset=["growth_percentage"]).sort_values("growth_percentage",ascending=False))




#GDP PER CAPITA__________________________________________________________________________________________
gdp_file = "DATASETS/worldbank_gdp_per_capita.csv.csv"
gdp = pd.read_csv(gdp_file,skiprows=4) 

#-->the metadata row from the csv file at the top are giving an error

print("\nGDP dataset columns:")
print(gdp.columns.tolist())

print("\nGDP first 5 rows:")
print(gdp.head())

print("\nGDP row count:")
print(len(gdp))

#extracting the required GDP 
gdp_2024 = gdp[["Country Code","Country Name","2024"]].copy()

gdp_2024 = gdp_2024.rename(
    columns= {
        "Country Code": "reporterISO",
        "Country Name": "gdp_country",
        "2024": "gdp_per_capita"
    }
)
print("\nGDP 2024 data:")
print(gdp_2024.head(20))


market_data = market_data.merge(gdp_2024[["reporterISO", "gdp_per_capita"]],
                                on="reporterISO",
                                how="left")

print("\nMarket Data with GDP:")
print(market_data[["reporterISO","reporterDesc","demand","growth_percentage","gdp_per_capita"]]
      .head(20).to_string(index=False))

complete_markets = market_data.dropna(
    subset=["demand","growth_percentage","gdp_per_capita"]
)

print("\nMarkets with demand + growth + GDP:")
print(len(complete_markets))

print("\nMarket data with GDP:")
print(market_data[["reporterISO","reporterDesc","demand","growth_percentage","gdp_per_capita"]]
      .dropna(subset=["demand"])
      .sort_values("demand",ascending=False)
      .head(15)
      .to_string(index=False))

#logistics peprformance index________________________________________________________________________________
lpi_file = "DATASETS/worldbank_logistics_lpi.csv.csv"
lpi = pd.read_csv(lpi_file, skiprows=4)

print("\nLPI columns:")
print(lpi.columns.tolist())

print("\nLPI first 5 rows:")
print(lpi.head())

print("\nLPI row count:")
print(len(lpi))

#REMOVE DUPLPICATES
print("\nLPI indicator:")
print(lpi[["Indicator Name", "Indicator Code"]].drop_duplicates())
print("\nLPI data coverage by recent year:")

for year in["2018","2019","2020","2021","2022","2023","2024","2025"]:
    available = lpi[year].notna().sum()
    print(year,":",available,"countries")

#identify the latest year that has LPI data 
recent_years = ["2018","2019","2020","2021","2022","2023","2024","2025"]
available_years = [year 
                   for year in recent_years
                   if lpi[year].notna().sum()>0]
latest_lpi_year = available_years[-1]
print("\nLatest available LPI year:", latest_lpi_year)

lpi_latest = lpi [["Country Code","Country Name",latest_lpi_year]].copy()

#renaming columns to match MArketAI
lpi_latest = lpi_latest.rename(
    columns={
    "Country Code": "reporterISO",
    "Country Name": "lpi_country",
    latest_lpi_year:"logistics_lpi"
})

print("\nLPI latest columns:")
print(lpi_latest.columns.tolist())

#merge 
market_data= market_data.merge(
    lpi_latest[["reporterISO","logistics_lpi"]],
                               on="reporterISO",
                               how="left")

print(market_data[[
    "reporterISO", "reporterDesc","demand","growth_percentage","gdp_per_capita","logistics_lpi"]]
    .dropna(subset=["demand"])
    .sort_values("demand",ascending=False)
    .head(15)
    .to_string(index=False))

print("\nLPI match summary:")
print("Markets with LPI:", market_data["logistics_lpi"].notna().sum())
print("Markets without LPI:", market_data["logistics_lpi"].isna().sum())

#RANKING ENGINE!!________________________________________________________________________________________________
ranking = market_data[["reporterISO", "reporterDesc","demand","growth_percentage","gdp_per_capita","logistics_lpi"]].dropna().copy()

print("\nMarkets available for ranking:",len (ranking))

#reducing the skewness,because the demand values can be large
ranking["demand_log"] =np.log1p(ranking["demand"]) #ln(1+x) 

#normalisation function
def normalize(series):
    maximum = series.max()
    minumum = series.min()

    if maximum == minumum:
        return pd.Series(0.5, index=series.index)
    return(
        (series - minumum) / (maximum - minumum)
    )

#normalizing each criterion 
ranking["demand_norm"] = normalize(ranking["demand_log"])
ranking["growth_norm"] = normalize(ranking["growth_percentage"])
ranking["gdp_norm"]= normalize(ranking["gdp_per_capita"])
ranking["logistics_norm"] = normalize(ranking["logistics_lpi"])

#WEIGHTED MARKET SCORE_____________________________________________________
ranking["market_score"] = (ranking["demand_norm"]*0.40 
                           + ranking["growth_norm"]*0.20
                           +ranking["gdp_norm"]*0.20
                           +ranking["logistics_norm"]*0.15)


#Convert score to 0-100 
ranking["market_score_100"]=(ranking["market_score"]*100)

#rank countries
ranking = ranking.sort_values("market_score",ascending=False)

ranking["rank"] = range(1,len(ranking)+1)
print("\nTop 10 marketAI recommendations:")

print(ranking[
    ["rank","reporterISO", "reporterDesc","demand","growth_percentage","gdp_per_capita","logistics_lpi","market_score_100"]].head(10)
    .to_string(index=False))

#save processed results
ranking.to_csv("DATASETS/marketai_ranked_furniture.csv",index=False)
print("\nSaved:DATASETS/marketai_ranked_furniture.csv")
