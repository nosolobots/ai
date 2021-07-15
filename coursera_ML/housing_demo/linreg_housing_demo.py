import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import linreg as reg

# Load data
df = pd.read_csv('data/housing.data', header=None, delim_whitespace=True)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# Create X from CSV in numpy
X = df.drop(columns=['MEDV']).values  # drop y column and convert to np array
X = np.c_[np.ones(X.shape[0]), X]     # add ones's column

# Create y vector from CSV
y = df.MEDV.values

# normalize X
reg.norm_m(X, np.arange(1, X.shape[1]))

# normalize y
y = reg.norm_v(y)

# Gradien descent
alpha = 0.1
num_iter = 1000
theta = np.ones(X.shape[1])
theta_list = reg.gradient_descent(theta, X, y, num_iter, alpha)

# Calculate cost for theta_list
j = []
for t in theta_list:
    j.append(reg.cost(t, X, y))

print("---> Gradient descent (", num_iter, "iterations)")
print("\ttheta:", theta_list[-1])
print("\tcost:", reg.cost(theta_list[-1], X, y))

# plot gradient descent
x = np.arange(num_iter)
plt.plot(x, j, 'b')
plt.xlabel("num iter")
plt.ylabel("J")
plt.show()

# Normal equation
theta = reg.normal_eq(X, y)
print("---> Normal Equation")
print("\ttheta:", theta)
print("\tcost:", reg.cost(theta, X, y))

# Plot hypothesis for calculated theta and some samples
x_test = np.arange(0, X.shape[0], 15) # one of every 15 samples
y_test = []
h_test = []
for i in x_test:
    y_test.append(y[i])
    h_test.append(reg.hypothesis(theta, X[i]))

plt.plot(x_test, y_test, 'ro', x_test, h_test, 'b')
plt.xlabel("samples")
plt.ylabel("y")
plt.legend(['y', 'hypothesis'])
plt.show()