import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
import pickle

df = pd.read_csv(r"C:\Training\Kaggle\Datasets\Heart Failure\heart.csv")

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

### Using One Hot Encoder

ohc = OneHotEncoder(handle_unknown='ignore')
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
                             ("passthrough",
                              make_column_selector(dtype_include=['int64','float64'])))
dum_np = ct.fit_transform(X)

# pickling_on = open(r"C:\Training\Kaggle\Datasets\Heart Failure\ct.pickle","wb")
# pickle.dump(ct, pickling_on)
# pickling_on.close()


dum_X = pd.DataFrame(dum_np,columns=ct.get_feature_names_out())
print(ct.get_feature_names_out())



parameters = {'max_features': np.arange(1,11)}

kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)

model_rf = RandomForestClassifier(random_state=2022)
cv = GridSearchCV(model_rf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc')

cv.fit( dum_X , y )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)

best_model = cv.best_estimator_

pickling_on = open(r"C:\Training\Kaggle\Datasets\Heart Failure\rf.pickle","wb")
pickle.dump(best_model, pickling_on)
pickling_on.close()


##########################Feature Importance########################

features = ct.get_feature_names_out()
importances = best_model.feature_importances_
indices = np.argsort(importances)

############ sorted #############
features = ct.get_feature_names_out()
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importance Plot')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()