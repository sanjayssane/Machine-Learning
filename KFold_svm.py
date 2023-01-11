import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyp = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyp, drop_first=True)

X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

svm = SVC(probability=True, random_state=2023, kernel="linear")

results = cross_val_score(svm, X, y, verbose=3,
                          scoring='roc_auc', cv=kfold)
print(results.mean())
