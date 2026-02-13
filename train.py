import os
import json
import joblib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------
# 1️⃣ Create Required Directories
# ---------------------------------------------------
os.makedirs("dataset", exist_ok=True)
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/evaluation", exist_ok=True)

# ---------------------------------------------------
# 2️⃣ Download Dataset if Not Present
# ---------------------------------------------------
dataset_path = "dataset/winequality-white.csv"

if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    urllib.request.urlretrieve(url, dataset_path)
    print("Download complete!")

# ---------------------------------------------------
# 3️⃣ Load Dataset
# ---------------------------------------------------
df = pd.read_csv(dataset_path, sep=";")

# Feature / target split
X = df.drop("quality", axis=1)
y = df["quality"]

# Select top correlated features
selected_features = df.corr()["quality"].abs().sort_values(ascending=False)[1:8].index
X = df[selected_features]

# ---------------------------------------------------
# 4️⃣ Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------
# 5️⃣ Model Training
# ---------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# 6️⃣ Evaluation
# ---------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Name: PARTHIV REDDY PELLURU")
print("Roll No: 2022BCS0073")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# ---------------------------------------------------
# 7️⃣ Save Model
# ---------------------------------------------------
joblib.dump(model, "output/model/model.pkl")

# ---------------------------------------------------
# 8️⃣ Save Metrics (VERY IMPORTANT FOR JENKINS)
# ---------------------------------------------------
results = {
    "accuracy": float(r2),   # Jenkins expects 'accuracy'
    "mean_squared_error": float(mse),
    "r2_score": float(r2)
}

with open("output/evaluation/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Training Completed Successfully!")
