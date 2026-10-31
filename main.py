from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger('FastAPI Iris Predictor')
try:
    model = joblib.load('model.pkl')
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error('Failed to load the model: %s', e)
    raise RuntimeError('Model loaded failed') from e

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
    try:
        with open('static/index.html', 'r') as file:
            logger.info('Home page served.')
            return file.read()
    except Exception as e:
        logger.error('Error serving home page: %s', e)
        return HTMLResponse(content = 'Error loading the home page.', status_code= 500)

@app.get('/predict', response_class=HTMLResponse)
async def predict():
    try:
        with open('static/predict.html', 'r') as file:
            logger.info('Prediction page served.')
            return file.read()
    except Exception as e:
        logger.error('Error serving prediction page: %s', e)
        return HTMLResponse(content = 'Error loading prediction page.', status_code= 500)

@app.post('/predict')
async def predict_species(
    SepalLengthCm: float = Form(...),
    SepalWidthCm: float = Form(...),
    PetalLengthCm: float = Form(...),
    PetalWidthCm: float = Form(...)):
    
    
    feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    try:
        input_df = pd.DataFrame([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]], columns = feature_names)
        logger.info('Input data recieved: %s', input_df)
        prediction = model.predict(input_df)[0]
        species = species_mapping.get(prediction, 'Unknown')
        logger.info('Prediction made: %s', species)
        return {'prediction': species}
    except Exception as e:
        logger.error('Error making prediction: %s', e)
        return JSONResponse(content = {'error': 'Failed to make a Prediction'}, status_code= 500)
