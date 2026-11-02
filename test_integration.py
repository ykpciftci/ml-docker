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

VALID_FLOWER = {'SepalLengthCm': 5.1, 'SepalWidthCm': 3.5, 'PetalLengthCm': 1.4, 'PetalWidthCm': 0.2}


def test_predict_returns_species():
    response = client.post('/predict', json = VALID_FLOWER)
    assert response.status_code == 200
    assert response.json() == {'prediction': 'Iris Setosa'}


def test_predict_accepts_upper_limit():
    response = client.post('/predict', json = {**VALID_FLOWER, 'SepalLengthCm': 15})
    assert response.status_code == 200


@pytest.mark.parametrize('field, bad_value', [
    ('SepalLengthCm', -1),
    ('SepalWidthCm', 0),  # zero
    ('PetalLengthCm', 16),  # above the limit
    ('PetalWidthCm', 'cm')  # not a number
])
def test_predict_rejects_invalid_values(field, bad_value):
    response = client.post('/predict', json = {**VALID_FLOWER, field: bad_value})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    flower = {**VALID_FLOWER}
    del flower['PetalWidthCm']
    response = client.post('/predict', json = flower)
    assert response.status_code == 422

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200, 'Failed to check health'
    assert response.json() == {'status': 'ok'} , 'Response missing JSON body'