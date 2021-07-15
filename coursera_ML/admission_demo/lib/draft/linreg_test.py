import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def norm_m(X, cols):
    # Normaliza la matriz
    # normaliza todos los valores de la columna: x = (x - mu)/range(x)
    for c in cols:
        X[:, c] = norm_v(X[:, c])


def norm_v(v):
    return (v - v.mean())/(v.max() - v.min())

def hypothesis(theta, X):
    m = X.shape[0]
    n = len(theta)
    hypo = np.zeros(m)
    for i in range(m):
        h = 0
        for j in range(n):
            h += theta[j]*X[i][j]
        hypo[i] = h
    
    return hypo

def vhypothesis(theta, X):
    return X@theta

def cost(theta, X, y):
    cost = 0
    m = X.shape[0]
    n = X.shape[1]

    for i in range(m):
        hypo = 0
        for j in range(n):
            hypo += theta[j]*X[i][j]
        cost += (hypo - y[i])**2
    cost *= (1/(2*m))

    return cost

def vcost(theta, X, y):
    m = X.shape[0]

    # alt 1
    #cost = (1/(2*m))*((X@theta - y)**2).sum()
    
    # alt 2
    cost = (1/(2*m))*np.transpose(X@theta - y)@(X@theta - y)

    return cost

def gradient(theta, X, y, alpha):
    m = X.shape[0]
    n = X.shape[1]

    new_theta = np.array(theta)

    for j in range(n):
        suma = 0
        for i in range(m):
            hypo = 0
            for k in range(n):
                hypo += theta[k]*X[i][k]
            suma += (hypo - y[i])*X[i][j]
        new_theta[j] = theta[j] - (alpha/m)*suma

    return np.array(new_theta)

def vgradient(theta, X, y, alpha):
    m = X.shape[0]
    n = X.shape[1]

    new_theta = theta - (alpha/m)*(X.T@(X@theta - y))

    return new_theta

def normal_eq(theta, X, y):
    return np.linalg.pinv(X.T@X)@X.T@y

# Load data
df = pd.read_csv('data/housing.data', header=None, delim_whitespace=True)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# Create X matrix
Xd = df.drop(columns=['MEDV'])
Xd.insert(0, 'X0', 1)
print(Xd.head())

# numpy array format
X = Xd.values
y = df.MEDV.values

# alternatively: create X from CSV in numpy
# X = df.drop(columns=['MEDV']).values  # drop y column and convert to np array
# X = np.c_[np.ones(X.shape[0]), X]     # add ones's column

# sample size
m = len(y)
print("\nm =", m)

# feature size
n = len(X[0])
print("\nn =", n)

# normalize X
norm_m(X, np.arange(1, X.shape[1]))

print(X)

# normalize y
y = norm_v(y)
print(y)

# theta
theta = np.ones(n)

#print(X[0])
#print(theta)

# Calculate hypothesis
#print("theta:", theta)
#print("hypothesis:", hypothesis(theta, X))

#print("theta:", theta)
#print("hypothesis (vector):", vhypothesis(theta, X))

# Calculate cost
print("theta:", theta)
print("cost:", cost(theta, X, y))
print("cost (vector):", vcost(theta, X, y))

# Gradien descent
"""
J_list = []
alpha = 0.1
num_iter = 100
theta_list = np.ones((num_iter, n))
J_list.append(vcost(theta_list[0], X, y))
for i in range(1, num_iter):
    theta_list[i] = gradient(theta_list[i-1], X, y, alpha)
    J_list.append(vcost(theta_list[i], X, y))
"""

# Gradient descent vectorized
J_list_v = []
alpha = 0.1
num_iter = 1000
theta_list_v = np.ones((num_iter, n))
J_list_v.append(vcost(theta_list_v[0], X, y))
for i in range(1, num_iter):
    theta_list_v[i] = vgradient(theta_list_v[i-1], X, y, alpha)
    J_list_v.append(vcost(theta_list_v[i], X, y))

print("Gradient Descent (",num_iter, "iterations)")
print("theta:", theta_list_v[-1])
print("cost:", J_list_v[-1])

x = np.arange(num_iter)
#plt.plot(x, J_list, 'r', J_list_v, 'b')
plt.plot(x, J_list_v, 'b')
plt.xlabel("iter")
plt.ylabel("J")
#plt.legend(['loop', 'vector'])
plt.show()

# Normal equation
theta = normal_eq(theta, X, y)
print("Normal Equation")
print("theta:", theta)
print("cost:", vcost(theta, X, y))

# Test hypothesis for calculated theta
x_test = np.arange(0, X.shape[0], 25)
y_test = []
h_test = []
for i in x_test:
    y_test.append(y[i])
    h_test.append(vhypothesis(theta, X[i]))
plt.plot(x_test, y_test, 'ro', x_test, h_test, 'bo')
plt.xlabel("samples")
plt.ylabel("y")
plt.legend(['y', 'hypothesis'])
plt.show()