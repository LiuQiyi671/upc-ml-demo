import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 简单练习
#    输出一个5*5的单位矩阵
# 1. Simple function
#    output a 5*5 identity matrix

A = np.eye(5)
print(A)

# 2. 单变量的线性回归
#    整个2的部分需要根据城市人口数量，预测开小吃店的利润。
#    数据在ex1data1.txt里，第一列是城市人口数量，第二列是该城市小吃店利润。
# 2. Linear regression with one variable
#    You will implement linear regression with one variable to
#    predict profits for a food truck.The file ex1data1.txt
#    contains the dataset for our linear regression problem.
#    The first column is the population of a city and the second
#    column is the profit of a food truck in that city.

# 2.1  展示数据
# 2.1  Plotting the data

data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', s=100, marker='x', c='red', x='Population', y='Profit')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.xticks(np.linspace(4, 24, 11))
plt.yticks(np.linspace(-5, 25, 7))
plt.show()


# 2.2  梯度下降
# 2.2  Gradient Descent

data.insert(0, 'ones', 1)
cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, cols - 1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0., 0.]))

def computeCost(X, y, theta):
    inner = np.power((X @ theta.T - y), 2)
    return np.sum(inner) / 2 / len(X)

print(computeCost(X,y,theta))



iters = 1500
alpha = 0.01

J_history = np.zeros(iters)

def gradientDescent(X, y, theta, alpha, iters):
    for i in range(iters):
        error = (X * theta.T - y)
        for j in range(theta.ravel().shape[1]):
            term = np.multiply(error, X[:, j])
            theta[0, j] = theta[0, j] - alpha / len(X) * np.sum(term)
        J_history[i] = computeCost(X, y, theta)
    return J_history, theta


J_history, theta = gradientDescent(X, y, theta, alpha, iters)
print(theta.shape)


