import joblib
import pandas as pd
from preprocess import preprocess_test
model = joblib.load("../models/XGBoost.pkl")

X_test = pd.read_csv("../data/test.csv") 
data = preprocess_test(X_test)
y_proba = model.predict_proba(data)[:, 1]


output = pd.DataFrame({
    "id": X_test["id"],
    "Exited": y_proba
})

output.to_csv("../predicts/predictions.csv", index=False)
print("✅ Saved predictions to predictions.csv")
