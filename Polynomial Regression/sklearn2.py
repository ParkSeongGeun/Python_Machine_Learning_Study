# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()

# PolynomialFeatures 함수를 통해 현재 데이터를 다항식 형태로 변경(다항 회귀를 위해 가공)
# degree 옵션으로 차수를 조절 (현재는 2차)
polynomial_transformer = PolynomialFeatures(degree = 2)
# 새롭게 정의된 numpy 배열 -> 기존 정보 + 기존 열 조합해 가상의 열을 추가하는 과정
polynomial_data = polynomial_transformer.fit_transform(diabetes_dataset.data)
# get_feature_names -> 만들어진 변수의 차수를 쉽게 확인 가능(속성 이름들을 받아옴)
polynomial_feature_names = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)
X = pd.DataFrame(polynomial_data, columns = polynomial_feature_names)

# 목표 변수
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

# 여기에 코드를 작성하세요
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)
model = LinearRegression()
model.fit(X_train, y_train)
y_test_prediction = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_prediction)
mse ** 0.5