import pandas as pd
from main import model

def test_model_prediction():
    input_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    species = model.predict(input_df)[0]
    assert species == 'Iris Setosa', 'The Prediction did not return the expected result.'