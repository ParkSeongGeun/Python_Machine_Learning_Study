from sklearn.datasets import load_boston
# data를 train set, test set 으로 나눠주는 함수
from sklearn.model_selection import train_test_split
# 선형회귀 함수
from sklearn.linear_model import LinearRegression
# 평균제곱오차 함수
from sklearn.metrics import mean_squared_error

import pandas as pd

boston_dataset = load_boston()
print(boston_dataset.DESCR)

boston_dataset.feature_names

boston_dataset.data

X = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

boston_dataset.target

y = pd.DataFrame (boston_dataset.target, columns = ['MEDV'])

# random_state에 값을 지정하는 것은 매번 실행할 때마다 동일한 data를 넘겨주기 위함
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)

# 학습
model = LinearRegression()
model.fit(X_train, y_train)
model.coef_
model.intercept_
y_test_prediction = model.predict(X_test)
y_test_prediction

# MSE
mean_squared_error(y_test, y_test_prediction)**0.5