"""Linear regression methods.

    - norm_m(X, cols):
        normalize matrix

    - norm_v(v):
        normalize vector (def: mean normalization)

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
    """
    for c in cols:
        X[:,c] = norm_v(X[:,c], t)

def hypothesis(theta, X):
    """Calculate the hypothesis.

    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features

    Returns:
        m-vector with the calculated hypothesis for every sample
    """
    return X@theta

def cost(theta, X, y):
    """Calculate the cost function.

    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results

    Returns:
        the calculated cost value
    """
    m = X.shape[0]  # number of samples

    # alternative method 1
    #return (1/(2*m))*((X@theta - y)**2).sum()

    # alternative method 2
    return (1/(2*m))*np.transpose(X@theta - y)@(X@theta - y)

def gradient(theta, X, y, alpha):
    """calculate the gradient for the cost function.

    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
        alpha: learning rate

    Returns:
        the calculated n-vector gradient
    """
    m = X.shape[0]  # number of samples
    return (alpha/m)*(X.T@(X@theta - y))

def gradient_descent(theta, X, y, niter, alpha):
    """Performs gradient descent a number of iterations

    Args:
        theta: n-vector of theta parameters
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
        niter: number of iterations
        alpha: learning rate

    Returns:
        (niter x n) matrix of calculated theta values
    """
    theta_list = np.zeros((niter, X.shape[1]))
    theta_list[0] = theta
    for i in range(1, niter):
        theta_list[i] = theta_list[i-1] - gradient(theta_list[i-1], X, y, alpha)

    return np.array(theta_list)

def normal_eq(X, y):
    """Calculate theta the optimized n-vector parameters using the normal equation

    Args:
        X: (m x n) matrix of m-samples x n-features.
        y: m-vector with the expected results
    Returns:
        the calculated theta parameters n-vector
    """
    return np.linalg.pinv(X.T@X)@X.T@y



