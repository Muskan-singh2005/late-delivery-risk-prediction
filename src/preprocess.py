import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    drop_cols = [
        'Customer Fname', 'Customer Lname', 'Customer Street',
        'Order City', 'Order State', 'Customer City',
        'Product Name',

        # 🚨 Remove leakage
        'Days for shipping (real)',
        'Delivery Status'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_data(df):
    # 🎯 Separate target BEFORE encoding
    target = df['Late_delivery_risk']
    features = df.drop('Late_delivery_risk', axis=1)

    features = pd.get_dummies(features, drop_first=True)

    return features, target