from itertools import product
from product_config import PRODUCTS

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
from pathlib import Path
from database import ( create_database, get_connection)


# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI(
    title="MarketAI",
    description="AI-assisted global market selection system",
    version="1.0"
)
create_database()
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)


# --------------------------------------------------
# FILE PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "DATASETS"
    / "marketai_final_data.csv"
)


# --------------------------------------------------
# LOAD MARKET DATA
# --------------------------------------------------

def load_market_data():

    return pd.read_csv( DATA_FILE )


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
# RANKING FUNCTION
# --------------------------------------------------
def rank_markets(
    weights,
    hs_code
):

    data_file = (
        BASE_DIR
        / "DATASETS"
        / f"marketai_final_{hs_code}.csv"
    )


    if not data_file.exists():

        raise FileNotFoundError(
            f"No market dataset for HS {hs_code}"
        )


    df = pd.read_csv(data_file )


    # -----------------------------------
    # AVAILABLE CRITERIA
    # -----------------------------------

    benefit_criteria = [
        "predicted_demand_2026",
        "predicted_growth_2026",
        "gdp_per_capita",
        "logistics_lpi",
        "political_stability"
    ]


    available_weights = {

        key: value

        for key, value
        in weights.items()

        if key in benefit_criteria

        or (
            key == "applied_tariff"
            and
            "applied_tariff" in df.columns
            and
            df[
                "applied_tariff"
            ].notna().any()
        )
    }


    # -----------------------------------
    # NORMALISE WEIGHTS
    # -----------------------------------

    total_weight = sum( available_weights.values())


    normalized_weights = {

        key:
            value / total_weight

        for key, value
        in available_weights.items()
    }


    # -----------------------------------
    # BENEFIT NORMALISATION
    # -----------------------------------

    for criterion in benefit_criteria:

        minimum = df[ criterion ].min()

        maximum = df[criterion ].max()


        if maximum == minimum:

            df[ criterion + "_norm" ] = 0.5

        else:

            df[ criterion + "_norm" ] = (
                df[criterion]
                -
                minimum
            ) / (
                maximum
                -
                minimum
            )


    # -----------------------------------
    # TARIFF COST NORMALISATION
    # -----------------------------------

    if ( "applied_tariff"  in available_weights ):

        minimum = df[ "applied_tariff" ].min()

        maximum = df[ "applied_tariff" ].max()


        if maximum == minimum:

            df["applied_tariff_norm" ] = 0.5

        else:

            df[ "applied_tariff_norm" ] = (
                maximum
                -
                df["applied_tariff"] ) / ( maximum - minimum )


    # -----------------------------------
    # SCORE
    # -----------------------------------

    df[ "market_score" ] = 0.0


    for criterion, weight in normalized_weights.items():
        contribution_column = (criterion + "_contribution")

    df[contribution_column] = (
        df[criterion + "_norm"]
        * weight
        * 100
    )

    df["market_score"] += ( df[contribution_column])

    return df.to_dict( orient="records" )
    # -----------------------------------
    # RANK
    # -----------------------------------

    df = df.sort_values(
        "market_score",
        ascending=False  )


    df["rank" ] = range(  1, len(df) + 1 )


    df[ "market_score" ] = df[ "market_score"].round(2)


    # -----------------------------------
    # TARIFF DISPLAY
    # -----------------------------------

    return df.to_dict( orient="records" )

# --------------------------------------------------
# WEBSITE ROUTES
# --------------------------------------------------

@app.get("/")
def home( request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/market-finder")
def market_finder(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="market_finder.html",
        context={}
    )


# --------------------------------------------------
# RESULTS ROUTE
# --------------------------------------------------

@app.post("/results")
def results(
    request: Request,
    company_name: str = Form(...),
    hs_code: str = Form(...),
    demand_weight: float = Form(...),
    growth_weight: float = Form(...),
    gdp_weight: float = Form(...),
    logistics_weight: float = Form(...),
    political_weight: float = Form(...),
    tariff_weight: float = Form(...)
):

    # -----------------------------------
    # PRODUCT
    # -----------------------------------

    product = PRODUCTS.get(hs_code)

    if product is None:

        return templates.TemplateResponse(
            request=request,
            name="market_finder.html",
            context={
                "error": "Selected product is not supported."
            }
        )


    product_name = product["display_name"]


    # -----------------------------------
    # WEIGHTS
    # -----------------------------------

    weights = {

        "predicted_demand_2026": demand_weight,

        "predicted_growth_2026":growth_weight,

        "gdp_per_capita":gdp_weight,

        "logistics_lpi":logistics_weight,

        "political_stability":political_weight,

        "applied_tariff":tariff_weight
    }


    if sum(weights.values()) <= 0:

        return templates.TemplateResponse(
            request=request,
            name="market_finder.html",
            context={
                "error":
                    "At least one weight must be greater than zero."
            }
        )


    # -----------------------------------
    # RANK MARKETS
    # -----------------------------------

    try:

        rankings = rank_markets(
            weights,
            hs_code
        )

    except FileNotFoundError:

        return templates.TemplateResponse(
            request=request,
            name="market_finder.html",
            context={
                "error":
                    f"Market analysis data for {product_name} "
                    "has not been generated yet."
            }
        )


    # -----------------------------------
    # RESULTS PAGE
    # -----------------------------------

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "company_name":company_name,

            "product_name": product_name,

            "hs_code":hs_code,

            "rankings":rankings,

            "weights": weights
        }
    )

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
@app.get("/dashboard")
def dashboard(request: Request):

    # Main 7-market dataset used by the WSM
    markets = pd.read_csv(
        BASE_DIR
        / "DATASETS"
        / "marketai_final_data.csv"
    )

    market_records = (
        markets
        .fillna(0)
        .to_dict(orient="records")
    )


    # AI evaluation dataset
    evaluation_file = (
        BASE_DIR
        / "DATASETS"
        / "model_evaluation_2025.csv"
    )

    if evaluation_file.exists():

        evaluation = pd.read_csv(
            evaluation_file
        ).head(10)

        evaluation_records = (
            evaluation
            .fillna(0)
            .to_dict(orient="records")
        )

    else:

        evaluation_records = []


    # WSM ranking dataset
    ranked_file = (
        BASE_DIR
        / "DATASETS"
        / "marketai_ranked_results.csv"
    )

    if ranked_file.exists():

        ranked = pd.read_csv( ranked_file )

        ranked_records = (
            ranked
            .fillna(0)
            .to_dict(orient="records") )

    else:

        ranked_records = []


    print(
        "Dashboard markets loaded:",
        len(market_records)
    )


    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "markets": market_records,
            "evaluation": evaluation_records,
            "ranked": ranked_records
        }
    )

# --------------------------------------------------
# MARKET COMPARISON ROUTE
# --------------------------------------------------
@app.get("/compare")
def compare_markets(request: Request):

    markets = pd.read_csv(
        BASE_DIR
        / "DATASETS"
        / "marketai_final_data.csv"
    )

    market_records = (
        markets
        .fillna(0)
        .to_dict(orient="records")
    )

    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "markets": market_records
        }
    )
# --------------------------------------------------
# METHODLOGY ROUTE
# --------------------------------------------------
@app.get("/methodology")
def methodology(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="methodology.html",
        context={}
    )
