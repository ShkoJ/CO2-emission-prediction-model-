
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Load dataset
df = pd.read_csv(r'C:/Users/shkob/Downloads/lab ST/CO2.csv')

# Define features and target
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']
target = 'CO2EMISSIONS'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Save model
joblib.dump(model, r'C:/Users/shkob/Downloads/lab ST/model.pkl')

# Calculate evaluation metrics
metrics = {
    "mean_squared_error": mean_squared_error(y_test, y_pred),
    "r2_score": r2_score(y_test, y_pred),
    "coefficients": model.coef_.tolist(),
    "intercept": model.intercept_
}

# Save metrics to a JSON file
with open(r"C:/Users/shkob/Downloads/lab ST/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Linear Regression model and metrics have been saved successfully.")
