from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
model = joblib.load('model.pkl')

app = FastAPI()

app.mount('/static', StaticFiles(directory = 'static'), name = 'static')

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

species_mapping = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}

@app.get('/', response_class= HTMLResponse)
async def home():
    with open('static/index.html', 'r') as file:
        return file.read()

@app.get('/predict', response_class=HTMLResponse)
async def predict():
    with open('static/predict.html', 'r') as file:
        return file.read()

@app.post('/predict')
async def predict_species(
    SepalLengthCm: float = Form(...),
    SepalWidthCm: float = Form(...),
    PetalLengthCm: float = Form(...),
    PetalWidthCm: float = Form(...)):
    
    
    feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    input_df = pd.DataFrame([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]], columns = feature_names)
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, 'Unknown')
    return {'prediction': species}
