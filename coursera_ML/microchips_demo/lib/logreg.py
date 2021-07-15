"""Linear regression methods.

    - norm_m(X, cols):
        normalize matrix

    - norm_v(v):
        normalize vector (def: mean normalization)

    - map_features(X1, X2, degree):
        feature mapping function to polynomial features

    - sigmoid(z):
        calculate the sigmoid funtion of the array

    - hypothesis(theta, X):
        calculate the hypothesis

    - cost(theta, X, y):
        calculate the cost function

    - gradient(theta, X, y, alpha):
        calculate the gradient for the cost function

    - gradient_descent(theta, X, y, niter, alpha)
        Performs gradient descent a number of iterations

    - normal_eq(theta, X, y):
        calculate theta the optimized n-vector parameters using the normal equation    
"""

import numpy as np

def norm_v(v, t='m'):
    """Normalize vector.

    Args:
        v: numpy array
        t: 'm' = mean normalization | 's' = std deviation normalization

    Returns:
        the normalized vector, mean, range or std
    """
    m = v.mean()
    
    if   t == 'm':  
        r = v.max() - v.min()
        return (v - m)/r, m, r
    elif t == 's':  
        s = v.std()
        return (v - m)/s, m, s

def norm_m(X, cols, t='m'):
    """Normalize matrix columns.

    Args:
        X: numpy vector matrix
        cols: list of columns to normalize
        type: 'm' = mean normalization | 's' = std deviation normalization

    Returns:
        the normalized matrix, mean vector, range or std vector
    """
    Xn = np.array(X)
    m = np.zeros(len(cols))
    rs = np.zeros(len(cols))

    for c in cols: 
        Xn[:,c], m[c], rs[c] = norm_v(Xn[:,c], t)
    
    return Xn, m, rs

def map_features(x1, x2, degree):
    """Feature mapping function to polynomial features.

    Maps the two input features to quadratic features. Returns a new feature array with more features,
    comprising of 1,x1, x2, x1², x2², x1*x2, x1*x2²,...

    Inputs x1, x2 must be the same size.

    Args: 
        X1: vector with feature 1
        X2: vector with feature 2
        degree: maximum polynomial power

    Returns:
        the feature matrix with on's column
    """
    M = np.ones(len(x1))    # first column
    for i in range(1, degree+1):
        for j in range(i+1):
            v = (x1**(i-j))*(x2**j)
            M = np.c_[M, v.T]

    return M

def sigmoid(z):
    """Calculate the sigmoid function.

    Args:
        z: numpy-vector

    Returns:
        numpy-vector with the calculated sigmoid value
    """ 
    return 1/(1 + np.e**(-z))

def hypothesis(theta, X):
    """Calculate the hypothesis.

    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features

    Returns:
        m-vector with the calculated hypothesis for every sample
    """ 
    return sigmoid(X@theta)        

def cost(theta, X, y, Lambda=0.0):
    """Calculate the cost function.

    Args:
        theta: (n,1) vector of theta parameters
        X: (m x n) matrix of m-samples x n-features
        y: (m,1) vector with the expected results
        Lambda: regularization parameter

    Returns:
        the calculated cost value
    """
    m = X.shape[0]  # number of samples

    h = hypothesis(theta, X)    
    
    suma = (-y*np.log(h) - (1 - y)*np.log(1 - h)).sum()    

    reg_term = 0.0
    if Lambda:
        reg_term = (Lambda/(2*m))*((theta[1:]**2).sum())    # skip theta-0

    return (1/m)*suma + reg_term

def gradient(theta, X, y, Lambda=0.0):
    """calculate the gradient for the cost function.
    
    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
        Lambda: regularization parameter

    Returns:
        the calculated n-vector gradient 
    """
    m = X.shape[0]  # number of samples

    h = hypothesis(theta, X)

    if Lambda:
        g_0 = (1/m)*(X.T@(h - y))[0]
        g_1 = (1/m)*(X.T@(h - y))[1:] + (Lambda/m)*theta[1:]    # skip theta-0
        
        return np.append(g_0, g_1)
    else:
        return (1/m)*(X.T@(h - y))

def gradient_descent(initial_theta, X, y, niter, alpha, Lambda=0.0):
    """Performs gradient descent a number of iterations

    Args:
        initial_theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
        niter: number of iterations
        alpha: learning rate
        Lambda: regularization parameter

    Returns:
        (niter x n) matrix of calculated theta values
        (niter) list of calculated cost values
    """
    theta_list = []
    cost_list = []

    theta = initial_theta
    for i in range(0, niter):
        theta -= alpha*gradient(theta, X, y, Lambda)
        theta_list.append(theta)
        cost_list.append(cost(theta, X, y, Lambda))

    return theta_list, cost_list

  
