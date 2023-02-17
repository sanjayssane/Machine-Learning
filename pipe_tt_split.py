import pandas as pd
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import os

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis")

kyp = pd.read_csv("Kyphosis.csv")
dum_kyp = pd.get_dummies(kyp, drop_first=True)

X = dum_kyp.drop('Kyphosis_present', axis=1)
y = dum_kyp['Kyphosis_present']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=2023)
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2023, kernel="rbf")
pipe_svm = Pipeline([('STD',scaler),('SVM', svm)])
pipe_svm.fit(X_train, y_train)

y_pred_prob = pipe_svm.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))


# X_trn_scl = scaler.fit_transform(X_train)  
# X_tst_scl = scaler.transform(X_test)


# svm.fit(X_trn_scl, y_train)

# y_pred_prob = svm.predict_proba(X_tst_scl)[:,1]
# print(roc_auc_score(y_test, y_pred_prob))













