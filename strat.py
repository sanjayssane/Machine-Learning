import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
cancer = pd.read_csv("BreastCancer.csv")
print(cancer.columns)

y = cancer['Class']
X = cancer.drop(['Code', 'Class'], axis=1)

y.value_counts(normalize=True)*100

y.value_counts()


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=22)
y_train.value_counts(normalize=True)*100
y_test.value_counts(normalize=True)*100

y_train.value_counts()
y_test.value_counts()


