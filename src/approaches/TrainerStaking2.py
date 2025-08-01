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
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
vote_est = [
    ('gbc', XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, loss='log_loss')),


    # ('xgb', XGBClassifier(
    #     use_label_encoder=False, eval_metric='logloss', verbosity=0, n_estimators=100, learning_rate=0.1, max_depth=3)),

]
def train_models(X_train, X_test, y_train, y_test, random_state=0):
    logistic_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
            solver='liblinear',
            penalty='l1',
            C=0.1,
            class_weight='balanced',
            max_iter=200
        ))
    ])

    # 2. XGBoost với best params
    xgboost_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.2,
        subsample=0.7,
        colsample_bytree=0.5,
        scale_pos_weight=1
    )


    stack_model = StackingClassifier(
        estimators=[
            ('logistic', logistic_pipe),
            ('xgb', xgboost_model)
        ],
        final_estimator=LogisticRegression(),
        passthrough=False,
        cv=5,
        n_jobs=-1
    )

    stack_model.fit(X_train, y_train)

    y_pred = stack_model.predict(X_test)
    
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Stacking (Tuned Logistic + XGBoost)")
    plt.show()