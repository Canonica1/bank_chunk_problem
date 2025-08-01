import numpy as np
import pandas as pd
from tqdm import  tqdm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.gaussian_process import GaussianProcessClassifier
import os
import joblib
from sklearn.linear_model import (
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    Perceptron
)
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier

vote_est = [
    ('gbc', XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, loss='log_loss')),
]
def train_models(X_train, X_test, y_train, y_test, random_state=0):
    cv_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=random_state)

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 7),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.8, 0.2),
        'colsample_bytree': uniform(0.8, 0.2),
        'reg_alpha': [0, 1],
        'reg_lambda': [1, 5, 10]
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        verbosity=1,
        random_state=random_state,
        
    )

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        scoring='roc_auc',
        cv=cv_split,
        n_iter=30,
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
        random_state=random_state
        
    )

    random_search.fit(X_train, y_train)

    print("Best parameters:", random_search.best_params_)
    print("Best ROC AUC: {:.2f}".format(random_search.best_score_ * 100))
    os.makedirs('../models', exist_ok=True)
    name = "xgb_final"
    model_filename = os.path.join('../models', f"{name.replace(' ', '_')}.pkl")
    joblib.dump(random_search.best_estimator_, model_filename)
    print(f"💾 Model saved to: {model_filename}")
    return name
