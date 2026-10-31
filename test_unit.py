import pytest
import pandas as pd
from main import model, species_mapping

def test_model_prediction():
    input_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, 'Unknown')
    assert species == 'Iris Setosa', 'The Prediction did not return the expected result.'