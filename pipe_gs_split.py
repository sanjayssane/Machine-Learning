import pandas as pd
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyp = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyp, drop_first=True)

X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']

scaler = StandardScaler()
svm = SVC(probability=True, random_state=2023, kernel="rbf")
pipe_svm = Pipeline([('STD',scaler),('SVM', svm)])

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=2023)
params = {'SVM__C':[0.001, 0.01, 0.5, 2, 3],
          'SVM__gamma': [0.01, 0.5, 2]}
gcv = GridSearchCV(pipe_svm, param_grid=params, scoring='roc_auc',
                  cv=kfold)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)


