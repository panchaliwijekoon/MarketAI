from product_config import PRODUCTS

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
from pathlib import Path
import os
import bcrypt

from database import (create_database, get_connection)

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI(
    title="MarketAI",
    description="AI-assisted global market selection system",
    version="1.0"
)
SECRET_KEY = os.getenv(
    "MARKETAI_SECRET_KEY",
    "marketai-development-key-change-this"
)


app.add_middleware(
    SessionMiddleware,

    secret_key=SECRET_KEY,

    max_age=3600,

    same_site="lax",

    https_only=False
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

PROFILE_PRODUCTS = {

    "940360": {
        "product_name": "Wooden Furniture",
        "industry": "Furniture"   },

    "2101": {
        "product_name": "Tea & Coffee Extracts / Preparations",
        "industry": "Food & Beverage"   },

    "420310": {
        "product_name": "Leather Apparel",
        "industry": "Apparel"   },

    "71": {
        "product_name": "Jewellery, Precious Metals & Stones",
        "industry": "Jewellery"    }

}

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

    data_file = (BASE_DIR / "DATASETS"/ f"marketai_final_{hs_code}.csv"  )


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

    if total_weight <= 0:

        raise ValueError(
            "No usable ranking weights were selected."
        )


    normalized_weights = {

        key: value / total_weight

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

    df["market_score"] = 0.0

    factor_names = {

        "predicted_demand_2026":
            "AI Predicted Demand",

        "predicted_growth_2026":
            "Predicted Growth",

        "gdp_per_capita":
            "GDP Per Capita",

        "logistics_lpi":
            "Logistics Performance",

        "political_stability":
            "Political Stability",

        "applied_tariff":
            "Tariff Accessibility"
    }


    for criterion, weight in normalized_weights.items():

        contribution_column = (
            criterion
            + "_contribution"
        )

        df[contribution_column] = (
            df[criterion + "_norm"]
            * weight
            * 100
        )

        df["market_score"] += (df[contribution_column] )


    # -----------------------------------
    # STRONGEST FACTOR
    # -----------------------------------

    def find_strongest_factor(row):

        strongest = max(
            normalized_weights.keys(),
            key=lambda criterion:
                row[
                    criterion
                    + "_contribution"
                ]
        )

        return factor_names.get(
            strongest,
            strongest
        )


    df["strongest_factor"] = ( df.apply( find_strongest_factor, axis=1  ) )


    # -----------------------------------
    # RANK
    # -----------------------------------

    df = df.sort_values(
        "market_score",
        ascending=False
    )

    df["rank"] = range(
        1,
        len(df) + 1
    )

    df["market_score"] = ( df["market_score"].round(2)  )


    # -----------------------------------
    # RETURN RESULTS
    # -----------------------------------

    return df.to_dict( orient="records")
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

#=======================================
# PROFILE ROUTE
#========================================
@app.get("/profile")
def profile_page(
    request: Request
):

    user = get_current_user( request )


    if user is None:

        return RedirectResponse(
            url="/login",
            status_code=303
        )


    connection = get_connection()


    profiles = connection.execute(
        """
        SELECT *
        FROM business_profiles

        WHERE user_id = ?

        ORDER BY id DESC
        """,

        ( user["id"],)
    ).fetchall()


    connection.close()


    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "profiles": profiles,
            "user": user
        }
    )

#=======================================
# PROFILE ROUTE --> business profile authentication
#========================================

def hash_password(
    password: str
):

    password_bytes = (
        password.encode("utf-8")
    )

    hashed = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


def verify_password(
    password: str,
    stored_hash: str
):

    return bcrypt.checkpw(

        password.encode("utf-8"),

        stored_hash.encode("utf-8")

    )
def get_current_user(
    request: Request
):

    user_id = request.session.get(
        "user_id"
    )


    if user_id is None:

        return None


    connection = get_connection()


    user = connection.execute(
        """
        SELECT id,
               full_name,
               email,
               created_at

        FROM users

        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()


    connection.close()


    return user

#=======================================
#REGISTER USER ROUTE
#========================================

@app.get("/register")
def register_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={}
    )



@app.post("/register")
def register_user(

    request: Request,

    full_name: str = Form(...),

    email: str = Form(...),

    password: str = Form(...),

    confirm_password: str = Form(...)
):

    email = ( email
        .strip()
        .lower()
    )


    # Basic password validation

    if len(password) < 8:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Password must contain at least 8 characters."
            }
        )


    if password != confirm_password:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Passwords do not match."
            }
        )


    connection = get_connection()


    existing_user = connection.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()


    if existing_user:

        connection.close()

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "An account with this email already exists."
            }
        )


    secure_password = hash_password(
        password
    )


    cursor = connection.execute(
        """
        INSERT INTO users
        (
            full_name,
            email,
            password_hash
        )

        VALUES (?, ?, ?)
        """,

        (
            full_name.strip(),
            email,
            secure_password
        )
    )


    connection.commit()


    user_id = cursor.lastrowid


    connection.close()


    # Automatically log them in

    request.session["user_id" ] = user_id


    return RedirectResponse(
        url="/profile",
        status_code=303
    )

#Log in
@app.get("/login")
def login_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={}
    )

@app.post("/login")
def login_user(

    request: Request,

    email: str = Form(...),

    password: str = Form(...)
):

    email = (
        email
        .strip()
        .lower()
    )


    connection = get_connection()


    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()


    connection.close()


    # Generic error prevents revealing
    # whether an email exists.

    if (
        user is None
        or not verify_password(
            password,
            user["password_hash"]
        )
    ):

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                    "Invalid email or password."
            }
        )


    # Prevent old session information
    # remaining after login.

    request.session.clear()


    request.session[ "user_id" ] = user["id"]


    return RedirectResponse(
        url="/profile",
        status_code=303
    )
@app.post("/logout")
def logout(
    request: Request
):

    request.session.clear()


    return RedirectResponse(
        url="/login",
        status_code=303
    )

@app.post("/profile")
def create_profile(
    request: Request,
    company_name: str = Form(...),
    hs_code: str = Form(...),
    export_capacity: float = Form(...),
    preferred_region: str = Form(...),
    experience_level: str = Form(...),
    contact_email: str = Form(...)
):

    # -----------------------------------
    # CHECK LOGGED-IN USER
    # -----------------------------------

    user = get_current_user(
        request
    )

    if user is None:

        return RedirectResponse(
            url="/login",
            status_code=303
        )


    # -----------------------------------
    # GET PRODUCT DETAILS
    # -----------------------------------

    product = PROFILE_PRODUCTS.get(
        hs_code
    )

    if product is None:

        connection = get_connection()

        profiles = connection.execute(
            """
            SELECT *
            FROM business_profiles
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user["id"],)
        ).fetchall()

        connection.close()

        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={
                "profiles": profiles,
                "user": user,
                "error":
                    "Unsupported product selected."
            }
        )


    # -----------------------------------
    # SAVE PROFILE
    # -----------------------------------

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO business_profiles
        (
            user_id,
            company_name,
            industry,
            product_name,
            hs_code,
            export_capacity,
            preferred_region,
            experience_level,
            contact_email
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (
            user["id"],
            company_name,
            product["industry"],
            product["product_name"],
            hs_code,
            export_capacity,
            preferred_region,
            experience_level,
            contact_email
        )
    )

    connection.commit()

    connection.close()

    return RedirectResponse(
        url="/profile",
        status_code=303
    )