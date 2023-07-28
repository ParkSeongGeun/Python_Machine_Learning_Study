import pandas as pd

TITANIC_FILE_PATH = '/Users/parkseonggeun/Desktop/Machine_Learning/Feature_scaling/titanic.csv'
titanic_df = pd.read_csv(TITANIC_FILE_PATH)
titanic_df.head()

titanic_sex_embarked = titanic_df[['Sex', 'Embarked']]
titanic_sex_embarked.head()

one_hot_encoded_df = pd.get_dummies(titanic_sex_embarked)
one_hot_encoded_df.head()

one_hot_encoded_df = pd.get_dummies(data=titanic_df, columns=['Sex', 'Embarked'])
one_hot_encoded_df.head()