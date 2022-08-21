import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv("Boston.csv")

y = df["medv"]
X = df.drop(["medv"],axis=1)

regressor = LinearRegression()
regressor.fit(X, y)
print(regressor.coef_)
print(regressor.intercept_)


############################ Train Test Split ############################

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2022)


regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

y_pred = regressor.predict(X_test)


print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

## Tree

regressor = DecisionTreeRegressor(random_state=222)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))





