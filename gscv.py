import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyp = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyp, drop_first=True)

X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

svm = SVC(probability=True, random_state=2023, kernel="rbf")
print(svm.get_params())
params = {'C':[0.001, 0.01, 0.5, 2, 3],
          'gamma': [0.01, 0.5, 2]}
gcv = GridSearchCV(svm, param_grid=params, scoring='roc_auc',
                  cv=kfold)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)
