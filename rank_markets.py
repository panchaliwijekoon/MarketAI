import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATASETS"

INPUT_FILE = (
    DATA_DIR / "marketai_final_data.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "marketai_ranked_results.csv"
)


# --------------------------------------------------
# LOAD FINAL MARKET DATA
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print("\n==============================")
print("MARKETAI WSM RANKING")
print("==============================")


# --------------------------------------------------
# USER / DEFAULT CRITERION WEIGHTS
# --------------------------------------------------

weights = {

    "predicted_demand_2026": 30,

    "predicted_growth_2026": 20,

    "gdp_per_capita": 10,

    "logistics_lpi": 15,

    "political_stability": 10,

    "applied_tariff": 15
}


# --------------------------------------------------
# NORMALISE WEIGHTS
# --------------------------------------------------

total_weight = sum(
    weights.values()
)


normalized_weights = {

    criterion:
        weight / total_weight

    for criterion, weight
    in weights.items()
}


print("\nNormalized weights:")

for criterion, weight in normalized_weights.items():

    print(
        criterion,
        "=",
        round(weight, 3)
    )


# --------------------------------------------------
# NORMALISATION FUNCTIONS
# --------------------------------------------------

def normalize_benefit(
    value,
    minimum,
    maximum
):

    if maximum == minimum:
        return 0.5

    return (
        value - minimum
    ) / (
        maximum - minimum
    )


def normalize_cost(
    value,
    minimum,
    maximum
):

    if maximum == minimum:
        return 0.5

    return (
        maximum - value
    ) / (
        maximum - minimum
    )


# --------------------------------------------------
# DEFINE CRITERION TYPES
# --------------------------------------------------

benefit_criteria = [

    "predicted_demand_2026",

    "predicted_growth_2026",

    "gdp_per_capita",

    "logistics_lpi",

    "political_stability"
]


cost_criteria = [

    "applied_tariff"
]


# --------------------------------------------------
# NORMALISE BENEFIT CRITERIA
# --------------------------------------------------

for criterion in benefit_criteria:

    minimum = df[
        criterion
    ].min()

    maximum = df[
        criterion
    ].max()


    df[
        criterion + "_norm"
    ] = df[
        criterion
    ].apply(

        lambda value:

        normalize_benefit(
            value,
            minimum,
            maximum
        )
    )


# --------------------------------------------------
# NORMALISE COST CRITERIA
# --------------------------------------------------

for criterion in cost_criteria:

    minimum = df[
        criterion
    ].min()

    maximum = df[
        criterion
    ].max()


    df[
        criterion + "_norm"
    ] = df[
        criterion
    ].apply(

        lambda value:

        normalize_cost(
            value,
            minimum,
            maximum
        )
    )


# --------------------------------------------------
# CALCULATE CONTRIBUTIONS
# --------------------------------------------------

for criterion in weights:

    contribution_column = (
        criterion
        + "_contribution"
    )


    df[
        contribution_column
    ] = (

        df[
            criterion + "_norm"
        ]

        * normalized_weights[
            criterion
        ]

        * 100
    )


# --------------------------------------------------
# CALCULATE FINAL WSM SCORE
# --------------------------------------------------

contribution_columns = [

    criterion
    + "_contribution"

    for criterion
    in weights
]


df[
    "market_score"
] = df[
    contribution_columns
].sum(
    axis=1
)


# --------------------------------------------------
# IDENTIFY STRONGEST FACTOR
# --------------------------------------------------

def strongest_factor(row):

    contributions = {

        criterion:
            row[
                criterion
                + "_contribution"
            ]

        for criterion
        in weights
    }


    return max(
        contributions,
        key=contributions.get
    )


df[
    "strongest_factor"
] = df.apply(
    strongest_factor,
    axis=1
)


# --------------------------------------------------
# SORT MARKET RANKINGS
# --------------------------------------------------

df = df.sort_values(
    "market_score",
    ascending=False
)


# --------------------------------------------------
# ADD RANK
# --------------------------------------------------

df[
    "rank"
] = range(
    1,
    len(df) + 1
)


# --------------------------------------------------
# ROUND DISPLAY VALUES
# --------------------------------------------------

df[
    "market_score"
] = df[
    "market_score"
].round(2)


for column in contribution_columns:

    df[column] = (
        df[column]
        .round(2)
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY RANKING
# --------------------------------------------------

print("\n==============================")
print("FINAL MARKET RANKING")
print("==============================")

print(
    df[
        [
            "rank",
            "country",
            "market_score",
            "predicted_demand_2026",
            "predicted_growth_2026",
            "applied_tariff",
            "strongest_factor"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCreated:",
    OUTPUT_FILE.name
)