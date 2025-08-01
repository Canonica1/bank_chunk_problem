import numpy as np
import pandas as pd
from tqdm import  tqdm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    StackingClassifier,
    GradientBoostingClassifier,

)

import os
import joblib

from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline

def train_models(X_train, X_test, y_train, y_test, random_state=0):
    base_estimators = [
    ('gb',  GradientBoostingClassifier()),
    ('cb',  CatBoostClassifier(verbose=0)),
    ('lgb', LGBMClassifier(force_col_wise=True))
]
    meta_estimator = make_pipeline(
        StandardScaler(with_mean=False), 
        LogisticRegression(max_iter=1500)
    )
    
    stack = StackingClassifier(
        estimators       = base_estimators,
        final_estimator  = meta_estimator,
        cv               = 5,           
        n_jobs           = -1,    
        passthrough      = False              
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stack, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    stack.fit(X_train, y_train)
    print(f"Stacking ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    os.makedirs('../models', exist_ok=True)
    name = "xgb_final"
    model_filename = os.path.join('../models', f"{name.replace(' ', '_')}.pkl")
    joblib.dump(stack, model_filename)
    print(f"💾 Model saved to: {model_filename}")
    return name