import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get('/')
    assert response.status_code == 200, 'failed to load the page.'

def test_prediction_endpoint():
    response = client.get('/predict')
    assert response.status_code == 200, 'Failed to load the prediction page.'

def test_predict_species_endpoint():
    payload = {
        'SepalLengthCm' : 5.1,
        'SepalWidthCm' : 3.5,
        'PetalLengthCm': 1.4,
        'PetalWidthCm': 0.2
    }
    response = client.post('/predict', data = payload)
    assert response.status_code == 200, 'Failed to predict Species.'
    assert 'prediction' in response.json(), 'Response missing prediction field.'
