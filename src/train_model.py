import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

from preprocess import load_data, clean_data, encode_data
from feature_engineering import create_features


def train_pipeline():

    print("🚀 Training started...")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'supply_chain_data.csv')
    model_path = os.path.join(base_dir, '..', 'models', 'model.pkl')

    # =========================
    # 1. LOAD + PREPROCESS
    # =========================
    df = load_data(data_path)
    df = clean_data(df)
    df = create_features(df)

    # ✅ Correct encoding (fixed)
    X, y = encode_data(df)

    print("Dataset shape:", X.shape)

    # =========================
    # 2. TRAIN-TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 3. HANDLE IMBALANCE
    # =========================
    train_df = X_train.copy()
    train_df['target'] = y_train

    df_majority = train_df[train_df.target == 0]
    df_minority = train_df[train_df.target == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X_train = df_balanced.drop('target', axis=1)
    y_train = df_balanced['target']

    print("Balanced dataset shape:", X_train.shape)

    # =========================
    # 4. MODEL TRAINING
    # =========================
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("Model trained ✅")

    # =========================
    # 5. EVALUATION
    # =========================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

    print("\n📈 ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    print("\n📊 Average Probability:", y_prob.mean())

    # =========================
    # 6. FEATURE IMPORTANCE
    # =========================
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n🔥 Top 10 Important Features:\n", importance.head(10))

    # =========================
    # 7. SAVE MODEL
    # =========================
    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "columns": list(X.columns)
        }, f)

    print("\n✅ Model saved successfully at:", model_path)


if __name__ == "__main__":
    train_pipeline()