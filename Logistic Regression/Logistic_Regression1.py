from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

iris_data = load_iris()
X = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns = ['class'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()
model = LogisticRegression(solver='saga', max_iter=2000)
# 훈련 데이터(X_train)와 레이블(y_train)을 사용하여 모델 학습
model.fit(X_train, y_train)
# 테스트 데이터를 사용하여 예측 결과를 얻음
model.predict(X_test)
# 어느 정도로 분류를 잘하는 지 수치화
model.score(X_test, y_test)