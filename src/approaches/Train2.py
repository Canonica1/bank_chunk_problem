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
    RandomForestClassifier
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
MLA = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    # RandomForestClassifier(),

    # LogisticRegressionCV(max_iter=3000),
    # PassiveAggressiveClassifier(),
    # RidgeClassifierCV(),
    # SGDClassifier(),
    # Perceptron(),

    # KNeighborsClassifier(),

    # DecisionTreeClassifier(),
    # ExtraTreeClassifier(),

    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis(),

    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    LGBMClassifier(),
    CatBoostClassifier(verbose=0)
]

def train_models(X_train, X_test, y_train, y_test, random_state=0):

    cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, \
                             random_state=random_state)

    MLA_columns = [
        'MLA Name',
        'MLA Parameters',
        'MLA Train ROC AUC Mean',
        'MLA Test ROC AUC Mean',
        'MLA Test ROC AUC 3*STD',
        'MLA Time'
    ]
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    MLA_predict = pd.DataFrame(index=pd.Index(y_test.index, name='Index'))

    for idx, alg in enumerate(tqdm(MLA)):
        name = alg.__class__.__name__
        params = alg.get_params()

        try:
            cv_results = cross_validate(
                alg, X_train, y_train,
                cv=cv_split,
                return_train_score=True,
                scoring='roc_auc'
            )

            MLA_compare.loc[idx, 'MLA Name'] = name
            MLA_compare.loc[idx, 'MLA Time'] = cv_results['fit_time'].mean()
            MLA_compare.loc[idx, 'MLA Train ROC AUC Mean'] = cv_results['train_score'].mean()
            MLA_compare.loc[idx, 'MLA Test ROC AUC Mean'] = cv_results['test_score'].mean()
            MLA_compare.loc[idx, 'MLA Test ROC AUC 3*STD'] = cv_results['test_score'].std() * 3
            MLA_compare.loc[idx, 'MLA Zarameters'] = str(params)

            alg.fit(X_train, y_train)

            # Предсказание вероятностей для ROC AUC
            if hasattr(alg, "predict_proba"):
                y_score = alg.predict_proba(X_test)[:, 1]
            elif hasattr(alg, "decision_function"):
                y_score = alg.decision_function(X_test)
            else:
                continue  # невозможно посчитать ROC AUC

            MLA_predict[name] = y_score

        except Exception as e:
            print(f"Ошибка в модели {name}: {e}")

    MLA_compare.sort_values(
        by='MLA Test ROC AUC Mean', ascending=False, inplace=True
    )
    print(MLA_compare)
    print(MLA_predict)
    sns.barplot(x='MLA Train ROC AUC Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

    #prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algorithm')
    MLA_compare.to_csv("MLA_compare.csv", index=False)

    return MLA_compare, MLA_predict