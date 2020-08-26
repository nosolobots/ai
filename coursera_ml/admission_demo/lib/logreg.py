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
        The normalized vector
    """
    if   t == 'm':  
        return (v - v.mean())/(v.max() - v.min())
    elif t == 's':  
        return (v - v.mean())/v.std()

def norm_m(X, cols, t='m'):
    """Normalize matrix columns.

    Args:
        X: numpy vector matrix
        cols: list of columns to normalize
        type: 'm' = mean normalization | 's' = std deviation normalization

    Returns:
        the normalized matrix
    """
    Xn = np.array(X)
    for c in cols: 
        Xn[:,c] = norm_v(Xn[:,c], t)
    
    return Xn

def map_features(x1, x2, degree):
    """Feature mapping function to polynomial features.

    Maps the two input features to quadratic features. Returns a new feature array with more features,
    comprising of x1, x2, x1², x2², x1*x2, x1*x2²,...

    Inputs x1, x2 must be the same size.

    Args: 
        X1: vector with feature 1
        X2: vector with feature 2
        degree: maximum polynomial power

    Returns:
        the feature matrix
    """
    M = np.ones(len(x1))
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

def cost(theta, X, y):
    """Calculate the cost function.

    Args:
        theta: (n,1) vector of theta parameters
        X: (m x n) matrix of m-samples x n-features
        y: (m,1) vector with the expected results

    Returns:
        the calculated cost value
    """
    m = X.shape[0]  # number of samples

    h = hypothesis(theta, X)    
    
    suma = (-y*np.log(h) - (1 - y)*np.log(1 - h)).sum()    

    return (1/m)*suma
    
def gradient(theta, X, y):
    """calculate the gradient for the cost function.
    
    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results

    Returns:
        the calculated n-vector gradient 
    """
    m = X.shape[0]  # number of samples

    h = hypothesis(theta, X)

    return (1/m)*(X.T@(h - y))

def gradient_descent(initial_theta, X, y, niter, alpha):
    """Performs gradient descent a number of iterations

    Args:
        initial_theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
        niter: number of iterations
        alpha: learning rate

    Returns:
        (niter x n) matrix of calculated theta values
        (niter) list of calculated cost values
    """
    theta_list = []
    cost_list = []

    theta = initial_theta
    for i in range(0, niter):
        theta -= alpha*gradient(theta, X, y)
        theta_list.append(theta)
        cost_list.append(cost(theta, X, y))

    return theta_list, cost_list

  
