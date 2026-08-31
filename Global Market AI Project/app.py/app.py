from fastapi import FASTAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
import pandas as pd 
import json 

app = FASTAPI(title="MarketAI",description=" International Markets" )

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static",name="static"))

