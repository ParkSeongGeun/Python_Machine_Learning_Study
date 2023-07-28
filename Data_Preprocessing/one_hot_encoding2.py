import pandas as pd

GENDER_FILE_PATH = './datasets/gender.csv'

gender_df = pd.read_csv(GENDER_FILE_PATH)
input_data = gender_df.drop(['Gender'], axis=1)

X = pd.get_dummies(input_data)

X.head()