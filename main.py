from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
model = joblib.load('model.pkl')

app = FastAPI()

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

species_mapping = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}

@app.get('/')
async def home():
    """
    Home Endpoint to Check if the Server is Running.

    """
    return {'message': 'Welcome to Iris Species Predictor API!'}

@app.post('/predict')
async def predict_species(input_data: IrisInput):
    """
    Predict endpoint to classify the Iris species
    Accepts JSON input and returns the predicted species
    """
    input_array = np.array([[
        input_data.SepalLengthCm,
        input_data.SepalWidthCm,
        input_data.PetalLengthCm,
        input_data.PetalWidthCm ]])
    feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    input_df = pd.DataFrame([input_data.dict()], columns = feature_names)
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, 'Unknown')
    return {'prediction': species}
