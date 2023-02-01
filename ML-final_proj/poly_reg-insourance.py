import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

df = pd.read_excel('insurance.xlsx')


le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()

model.fit(X_poly, y)


new_person = [[25, 0, 28, 0, 1, 2]]
new_person_poly = poly.fit_transform(new_person)
prediction = model.predict(new_person_poly)
print("Predicted charges for the new person: ", prediction)


plt.scatter(X_poly[:, 1], y, color='blue')
plt.plot(X_poly[:, 1], model.predict(X_poly), color='red')
plt.title('polynomial regression')
plt.xlabel('age')
plt.ylabel('charges')
plt.show()