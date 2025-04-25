# main.py

from churn_predictor import load_data, preprocess_data, train_model, evaluate_model, save_model

# Step 1: Load data
df = load_data("data/telco_churn.csv")

# Step 2: Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Step 3: Train model
model = train_model(X_train, y_train)

# Step 4: Evaluate model
evaluate_model(model, X_test, y_test)

# Step 5: Save model
save_model(model, "churn_model.pkl")



# ----------- Extra: Predict for a new customer -----------

import joblib

# Load the saved model
model = joblib.load("churn_model.pkl")

# Load and preprocess again to get the feature columns
from churn_predictor import load_data, preprocess_data
df = load_data("data/telco_churn.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

# Pick one customer from test set
new_customer = X_test.iloc[0:1]

# Predict
prediction = model.predict(new_customer)

print("New Customer Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
