import os
import pickle
import pandas as pd


def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'model.pkl')

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data["model"], data["columns"]


def preprocess_input(input_data, columns):
    df = pd.DataFrame([input_data])

    # Apply encoding
    df = pd.get_dummies(df)

    # 🚨 FIX: Align columns with training data
    df = df.reindex(columns=columns, fill_value=0)

    return df


def predict_risk(input_data):
    model, columns = load_model()

    df = preprocess_input(input_data, columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability