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
    dtree = DecisionTreeClassifier(random_state = 0)
    base_results = cross_validate(dtree, X_train, y_train,return_train_score=True, cv  = cv_split)
    dtree.fit(X_train, y_train)

    print('BEFORE DT Parameters: ', dtree.get_params())
    print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
    print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
    print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
    print('-'*10)


    param_grid = {'criterion': ['gini', 'entropy'],
                'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
                'random_state': [0] 
                }

    tune_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
    tune_model.fit(X_train, y_train)

    print('AFTER DT Parameters: ', tune_model.best_params_)
    print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
    print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
    print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
    print('-'*10)



    print('BEFORE DT RFE Training Shape Old: ', X_train.shape) 
    print('BEFORE DT RFE Training Columns Old: ', X_train.columns.values)

    print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
    print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
    print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
    print('-'*10)



    dtree_rfe = RFECV(dtree, step = 1, scoring = 'roc_auc', cv = cv_split)
    dtree_rfe.fit(X_train, y_train)

    X_rfe = X_train.columns.values[dtree_rfe.get_support()]
    rfe_results = cross_validate(dtree, X_train[X_rfe], y_train, return_train_score=True, cv  = cv_split)

    print('AFTER DT RFE Training Shape New: ', X_train[X_rfe].shape) 
    print('AFTER DT RFE Training Columns New: ', X_rfe)

    print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 
    print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))
    print('-'*10)


    #tune rfe model
    rfe_tune_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
    rfe_tune_model.fit(X_train[X_rfe], y_train)

    print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
    print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
    print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
    print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
    print('-'*10)