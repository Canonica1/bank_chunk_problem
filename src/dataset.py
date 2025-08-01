import pandas as pd

def load_train():
    
    data = pd.read_csv("../data/train.csv")
    return data