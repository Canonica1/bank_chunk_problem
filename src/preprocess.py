import pandas as pd
import numpy as np
from encoders import (
    KFoldTargetEncoderTrain,
    KFoldTargetEncoderTest
)


def preprocess_main(data):
    data['Age^2'] = data['Age'] ** 2
    data = data.drop(['CustomerId', 'Surname', 'id', 'GeoGender'], axis=1, errors='ignore')
    data['CreditScore_Balance'] = data['CreditScore'] * data['Balance']

    data = pd.get_dummies(data, columns=["Geography", "Gender"], drop_first=True)
    
    data.columns = data.columns.astype(str).str.replace(r"[\[\]\<\>\(\),]", '_', regex=True)
    return data





def preprocess_label(data):
    kf_surname = KFoldTargetEncoderTrain(
        colnames='Surname',
        targetName='Exited',
        n_fold=5,
        verbosity=True,
        discardOriginal_col=False
    )
    data = kf_surname.transform(data)

 

    return data
def preprocess(data):
    data = preprocess_main(data)
    y = data["Exited"]
    X = data.drop("Exited", axis=1) 
    return X, y
def preprocess_test(data, train):

    kf_surname_test = KFoldTargetEncoderTest(
        train=train,
        colNames='Surname',
        encodedName='Surname_Kfold_Target_Enc'
    )
    data = kf_surname_test.transform(data)
    data = preprocess_main(data)
    return data