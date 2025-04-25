import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Handle missing and non-numeric values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # If training (has 'Churn'), do full preprocessing
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df = pd.get_dummies(df, drop_first=True)

        # Use 'Churn' or 'Churn_Yes' depending on one-hot result
        if 'Churn_Yes' in df.columns:
            y = df['Churn_Yes']
            X = df.drop('Churn_Yes', axis=1)
        else:
            y = df['Churn']
            X = df.drop('Churn', axis=1)

        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    # If predicting (no 'Churn'), just return transformed data
    else:
        df = pd.get_dummies(df, drop_first=True)
        return None, df, None, None

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the feature column names used for training
    joblib.dump(X_train.columns.tolist(), "model_columns.pkl")

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def save_model(model, filename):
    joblib.dump(model, filename)

