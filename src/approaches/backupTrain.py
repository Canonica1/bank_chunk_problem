import os
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
from catboost import CatBoostClassifier

def train_models(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(solver='liblinear', class_weight='balanced'))
            ]),
            'params': {
                'model__penalty': ['l1', 'l2'],
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__max_iter': [100, 200, 500]
            }
        },
        # 'Random Forest': {
        #     'pipeline': Pipeline([
        #         ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        #     ]),
        #     'params': {
        #         'model__n_estimators': [100, 200, 500],
        #         'model__max_depth': [None, 10, 20, 30],
        #         'model__min_samples_split': [2, 5, 10],
        #         'model__min_samples_leaf': [1, 2, 4],
        #         'model__bootstrap': [True, False]
        #     }
        # },
        # 'CatBoost': {
        #     'pipeline': Pipeline([('model', CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced'))]),
        #     'params': {
        #         'model__iterations': [500, 800],
        #         'model__depth': [4, 6, 8],
        #         'model__learning_rate': [0.01, 0.05, 0.1],
        #         'model__l2_leaf_reg': [3, 5, 7]
        #     }
        # },
        'XGBoost': {
            'pipeline': Pipeline([
                ('model', XGBClassifier(scale_pos_weight=4.0, eval_metric='logloss', random_state=42))
            ]),
            'params': {
                'model__n_estimators': [900,1000,1100],
                'model__max_depth': list(range(3, 8)),
                'model__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                'model__subsample': [round(x, 2) for x in list(np.arange(0.6, 1.05, 0.1))],
                'model__colsample_bytree': [round(x, 2) for x in list(np.arange(0.6, 1.05, 0.1))],
                'model__scale_pos_weight': [1, 2, 3, 5, 10, 20]
            }
        }
    }
    results = {}

    for name, config in models.items():
        print(f"\n🔍 Tuning hyperparameters for: {name}")
        param_list = list(ParameterSampler(config['params'], n_iter=15, random_state=42))

        best_score = -1
        best_model = None
        best_params = None

        progress = tqdm(param_list, desc=f"Searching {name}")

        for params in progress:
            model = clone(config['pipeline'])
            model.set_params(**params)

            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = scores.mean()

            progress.set_postfix(score=mean_score)

            if mean_score > best_score:
                best_score = mean_score
                best_model = clone(model).fit(X_train, y_train)
                best_params = params


        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] 
        roc_auc = roc_auc_score(y_test, y_prob)
        print("ROC-AUC", roc_auc)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Best params for {name}: {best_params}")
        print(f"📈 Accuracy on test set: {acc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        model_filename = os.path.join('../models', f"{name.replace(' ', '_')}.pkl")
        joblib.dump(best_model, model_filename)
        print(f"💾 Model saved to: {model_filename}")

        results[name] = {
            'model': best_model,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'best_params': best_params,
            'model_path': model_filename
        }

        print("=" * 60)

    return results
