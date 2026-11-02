from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger('FastAPI Iris Predictor')
try:
    model = joblib.load(BASE_DIR / 'model.pkl')
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error('Failed to load the model: %s', e)
    raise RuntimeError('Failed to load the model') from e

app = FastAPI()

app.mount('/static', StaticFiles(directory = BASE_DIR / 'static'), name = 'static')


class IrisInput(BaseModel):
    SepalLengthCm: float = Field(gt=0, le=15, allow_inf_nan=False)
    SepalWidthCm: float = Field(gt=0, le=15, allow_inf_nan=False)
    PetalLengthCm: float = Field(gt=0, le=15, allow_inf_nan=False)
    PetalWidthCm: float = Field(gt=0, le=15, allow_inf_nan=False)

class PredictionOutput(BaseModel):
    prediction: str

species_mapping = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}

@app.get('/')
def home():
    return FileResponse(BASE_DIR / 'static' / 'index.html')


@app.get('/predict')
def predict():
    return FileResponse(BASE_DIR / 'static' / 'predict.html')
            
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictionOutput)
def predict_species(flower: IrisInput):
    try:
        input_df = pd.DataFrame([flower.model_dump()])
        logger.info('Input data received: %s', input_df)
        prediction = model.predict(input_df)[0]
        species = species_mapping.get(prediction, 'Unknown')
        logger.info('Prediction made: %s', species)
        return {'prediction': species}
    except Exception as e:
        logger.error('Error making prediction: %s', e)
        return JSONResponse(content = {'error': 'Failed to make a prediction'}, status_code= 500)
