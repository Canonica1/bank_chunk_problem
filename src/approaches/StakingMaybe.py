import os
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_models(X_train, X_test, y_train, y_test, save_dir='../models'):
    os.makedirs(save_dir, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Определяем пайплайны и параметры с префиксом 'model__'
    models = {
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
        },

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
        }
    }

    fitted = {}
    for name, cfg in models.items():
        print(f"\n🔍 Tuning hyperparameters for: {name}")
        param_list = list(ParameterSampler(cfg['params'], n_iter=10, random_state=42))
        best_score = -np.inf
        best_model = None
        progress = tqdm(param_list, desc=f"Searching {name}")
        for params in progress:
            pipe = clone(cfg['pipeline'])
            pipe.set_params(**params)
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model = clone(pipe).fit(X_train, y_train)
                best_params = params
            progress.set_postfix(score=scores.mean())
        # Оценка на тесте
        y_prob = best_model.predict_proba(X_test)[:,1]
        y_pred = best_model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} | ROC-AUC: {auc:.4f}, Acc: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(best_model, path)
        fitted[name] = best_model

    # Стекинг
    print("\n🔗 Training Stacking Ensemble")
    estimators = [(n, fitted[n]) for n in fitted]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(X_train, y_train)
    y_prob = stack.predict_proba(X_test)[:,1]
    y_pred = stack.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Stacking | ROC-AUC: {auc:.4f}, Acc: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    stack_path = os.path.join(save_dir, 'Stacking_Ensemble.pkl')
    joblib.dump(stack, stack_path)

    return fitted, stack
