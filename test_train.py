from train import build_model, load_data


def test_load_data_returns_all_flowers():
    X, y = load_data()
    assert len(X) == 150
    assert y.nunique() == 3


def test_model_learns_the_data():
    X, y = load_data()
    model = build_model()
    model.fit(X, y)
    assert model.score(X, y) > 0.9