from pathlib import Path
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
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


def train_model(X_train, y_train):
    model =make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma = 'scale', C = 1.0))
    model.fit(X_train, y_train)
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state= 42, stratify= y)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}')

    joblib.dump(model, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
