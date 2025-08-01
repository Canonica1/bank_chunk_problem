from dataset import load_train
from sklearn.model_selection import train_test_split
from preprocess import preprocess, preprocess_test, preprocess_label
from TrainerStaking import train_models
import pandas as pd
import joblib
def preprocess_surname(data):
    preds_csv_path = 'surname_predictions.csv'
    preds = pd.read_csv(preds_csv_path, header=None, names=["Surname", "raw_pred"])

    preds["pred_id"] = preds["raw_pred"].str.extract(r"\((\d+),")[0]
    preds = preds[["Surname", "pred_id"]]
    data['Surname_Coutry'] = pd.NA
    preds_dict = preds.set_index('Surname')["pred_id"].to_dict()
    for i in range(len(data)):
        data.at[i, "Surname_Coutry"] = int(preds_dict[data.at[i, "Surname"]])
    data["Surname_Coutry"] = data["Surname_Coutry"].astype('int')
    data.info()
    return data

data = load_train()
data_test = pd.read_csv('../data/test.csv')
data = preprocess_surname(data)
data_test = preprocess_surname(data_test)

data_with_label = preprocess_label(data)
X,y = preprocess(data_with_label)
test_X = preprocess_test(data_test,data_with_label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

name = train_models(X, X_test, y, y_test)


model = joblib.load(f"../models/{name}.pkl")

y_proba = model.predict_proba(test_X)[:, 1]


output = pd.DataFrame({
    "id": data_test["id"],
    "Exited": y_proba
})

output.to_csv("../predicts/predictions.csv", index=False)
print("✅ Saved predictions to predictions.csv")