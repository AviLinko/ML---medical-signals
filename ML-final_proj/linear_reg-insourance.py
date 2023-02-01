import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('insurance.xlsx')

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

model = LinearRegression()

model.fit(X, y)

new_person = [[25, 0, 28, 0, 1, 2]]
prediction = model.predict(new_person)
print("Predicted charges for the new person: ", prediction)