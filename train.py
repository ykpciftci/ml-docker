from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'Iris.csv'
MODEL_PATH = BASE_DIR / 'model.pkl'
FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def load_data():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data['Species'].str.replace('-', ' ').str.title()
    return X, y


def build_model():
    return make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma = 'scale', C = 1.0))

    
def evaluate(name, model, X, y):
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state = 42)
    scores = cross_val_score(model, X, y, cv = cv)
    print(f'{name}: {scores.mean():.4f} +/- {scores.std():.4f}')


def main():
    X, y = load_data()

    evaluate('Baseline', DummyClassifier(strategy = 'most_frequent'), X, y)
    evaluate('SVM', build_model(), X, y)

    model = build_model()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
