import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create output directories
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/evaluation", exist_ok=True)

# Load dataset (same logic as Lab 1)
df = pd.read_csv("dataset/winequality-white.csv", sep=";")

# Feature / target split
X = df.drop("quality", axis=1)
y = df["quality"]

selected_features = df.corr()["quality"].abs().sort_values(ascending=False)[1:8].index
X = df[selected_features]
y = df["quality"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model (start with Linear Regression)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (MANDATORY for Lab-2)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, "output/model/model.pkl")

# Save metrics
results = {
    "mean_squared_error": mse,
    "r2_score": r2
}

with open("output/evaluation/results.json", "w") as f:
    json.dump(results, f, indent=4)
