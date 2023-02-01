import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# getting the data
data = pd.read_excel("בדיקות_מעבדה.xlsx")
x = data.values[1:, 1]
y = data.values[1:, 2]
x_train, x_test = x[:40], x[40:]
y_train, y_test = y[:40], y[40:]


def calculate(theta, x, degree):
    sum = 0
    for i in range(degree):
        sum += x**i * theta[i]
    return sum

def loss(hx, y):
    L1 = np.mean(np.abs(y-hx))
    L2 = np.mean((y-hx)**2)
    print(f"L1: {L1}, L2: {L2}")


# polynomial 
degree = 4
poly = PolynomialFeatures(degree=degree)
reg = linear_model.LinearRegression()
reg.fit(poly.fit_transform(np.reshape(x_train, (-1, 1))), y_train)
slope1, intercept1 = reg.coef_, reg.intercept_
theta = np.concatenate((np.array([intercept1]),slope1[1:]), axis=0)

loss(calculate(theta, x_test, degree+1), y_test)

# drawing the graph
plt.title("ex3 hx")
plt.xlabel("H(x)")
plt.ylabel("y")
plt.scatter(calculate(theta, x_test, degree+1), y_test)
line = np.linspace(0,100)
plt.plot(line, line, "r")
plt.show()


# drawing the graph
plt.title("ex3")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x_test, y_test)
plt.plot(line, calculate(theta, line, degree+1), 'r')
plt.show()

# --------------------the linear regression we created in ex1 ---------------
slope2 = 0
intercept2 = 0
lr = 0.0005

# training the dataset
for i in range(100000):
    sum1 = 0
    sum2 = 0
    for j in range(len(x_train)):
        sum1 += (slope2*x_train[j] + intercept2) - y_train[j]
        sum2 += ((slope2*x_train[j] + intercept2) - y_train[j]) * x_train[j]
    intercept2 = intercept2 - lr * sum1/len(x)
    slope2 = slope2 - lr * sum2/len(x)

loss(calculate([intercept2,slope2],x_test,2), y_test)

# drawing the graph
plt.title("ex1 hx")
plt.xlabel("H(x)")
plt.ylabel("y")
plt.scatter(calculate([intercept2,slope2],x_test,2), y_test)
line = np.linspace(0, 100)
plt.plot(line , line, 'r')
plt.show()

# drawing the graph
plt.title("ex1")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x_test, y_test)
line = np.linspace(0, 100)
plt.plot(line , slope2*line+intercept2, 'r')
plt.show()