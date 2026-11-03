import pandas as pd
import pytest
from main import model

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


@pytest.mark.parametrize('measurements, expected', [
    ([5.1, 3.5, 1.4, 0.2], 'Iris Setosa'),
    ([5.9, 3.0, 4.2, 1.5], 'Iris Versicolor'),
    ([6.5, 3.0, 5.5, 2.0], 'Iris Virginica'),
])
def test_model_predicts_species(measurements, expected):
    input_df = pd.DataFrame([measurements], columns = FEATURES)
    assert model.predict(input_df)[0] == expected
