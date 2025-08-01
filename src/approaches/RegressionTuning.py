import numpy as np
import pandas as pd
from tqdm import  tqdm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import (
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    Perceptron
)

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from xgboost import XGBClassifier

def train_models(X_train, X_test, y_train, y_test, random_state=0):
    cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, \
                             random_state=random_state)
    logreg = LogisticRegressionCV(
    Cs=10,
    cv=cv_split,
    max_iter=3000,
    scoring='roc_auc',
    solver='lbfgs',
    random_state=random_state,
    n_jobs=-1,
)

    logreg.fit(X_train, y_train)

    logreg_results = cross_validate(
        logreg, X_train, y_train, cv=cv_split, return_train_score=True, scoring='roc_auc'
    )

    print('AFTER Logistic Regression CV')
    print("LogRegCV Training w/bin score mean: {:.2f}".format(logreg_results['train_score'].mean()*100))
    print("LogRegCV Test w/bin score mean: {:.2f}".format(logreg_results['test_score'].mean()*100))
    print("LogRegCV Test w/bin score 3*std: +/- {:.2f}".format(logreg_results['test_score'].std()*100*3))
    print('-'*10)