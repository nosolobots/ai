import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import lib.logreg as reg

def plot_data(X, y, show=True):
    # [x1, x2] where admission = 0
    neg = np.where(y==0)    # list of row-indices where y=0
    X_0 = X[neg]            # select samples where y=0

    # [x1, x2] where admission = 1
    pos = np.where(y==1)    # list of row-indices where y=1
    X_1 = X[pos]            # select samples where y=1

    plt.plot(X_0[:,1], X_0[:,2], 'bx')
    plt.plot(X_1[:,1], X_1[:,2], 'r+')

    plt.xlabel('Test 1')
    plt.ylabel('Test 2')

    if show:
        plt.legend(['QA not passed', 'QA passed'])
        plt.title('Input Data')
        plt.show()

def mapFeaturePlot(x1, x2, degree):
    """
    take in numpy array of x1 and x2, return all polynomial terms up to the given degree
    """
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j)) * (x2**j)
            out = np.hstack((out, terms))

    return out

def plot_decission_boundary(theta, X, y, degree):
    x1 = np.linspace(-1, 1.25, 50)
    x2 = np.linspace(-1, 1.25, 50)

    #x1 = np.linspace(X[:,1].min(), X[:,1].max(), 50)
    #x2 = np.linspace(X[:,2].min(), X[:,2].max(), 50)

    Z = np.zeros((len(x1),len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i,j] = mapFeaturePlot(x1[i], x2[j], degree)@theta

    cmap = plt.contourf(x1, x2, Z.T, levels=20)  # contour color map
    plt.contour(x1, x2, Z.T, 0, colors='k')

    # plot data
    plot_data(X, y, show=False)
    """
    neg = np.where(y==0)    # list of row-indices where y=0
    pos = np.where(y==1)    # list of row-indices where y=1
    plt.scatter(X[neg][:,1],X[neg][:,2],c="b",marker="x",label="QA not passed")
    plt.scatter(X[pos][:,1],X[pos][:,2],c="r",marker="+",label="QA Passed")
    """
    plt.legend(['QA not passed', 'QA Passed', 'Decision boundary'])
    plt.title('Logistic Regression (contour)')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    plt.colorbar(cmap)
    plt.show()    

def plot_gradient(num_iter, j_dict):
    x = np.arange(num_iter)

    for alpha, j_list in j_dict.items():
        plt.plot(x, j_list)

    plt.xlabel('No. iterations')
    plt.ylabel('Cost (J)')
    plt.title('Minimize cost function J\n(gradient descent)')

    alphas = list(j_dict.keys())
    plt.legend(['alpha: ' + str(a) for a in alphas])

    plt.show()

if __name__ == "__main__":
    # load data
    df = pd.read_csv('data/ex2data2.txt', header=None)
    df.columns = ['T1', 'T2', 'QA']

    # number of samples
    m = df.shape[0]

    # create X matrix
    X = df.drop(columns=['QA']).values
    X = np.c_[np.ones([m, 1]), X]

    # create y results
    y = df.QA.values

    # plot input data
    plot_data(X, y)

    # create X with new polynomical features from X1, X2
    degree = 6
    X = reg.map_features(df.T1.values, df.T2.values, degree)

    # numer of fetaures (including X0)
    n = X.shape[1]

    # cost and gradient test 1
    # theta = [0, 0, 0]
    # Lambda = 1
    # expected cost = 0.693
    # expected gradient = [0.0085 0.0188 0.0001 0.0503 0.0115...]
    theta = np.zeros(n)
    Lambda = 1

    print("------------------------------------------------")
    print("---> Test 1")
    print("theta:", theta)
    print("lambda:", Lambda)
    print("cost:", reg.cost(theta, X, y, Lambda), "\n(expected: 0.693)")

    print("grad [first 5]:", reg.gradient(theta, X, y, Lambda)[:5], 
                "\n(expected: [0.0085 0.0188 0.0001 0.0503 0.0115 ...])")

    # cost and gradient test 2
    # theta = np.ones(n)
    # expected cost = 3.16
    # expected gradient = [0.3460 0.1614 0.1948 0.2269 0.0922 ...]
    theta = np.ones(n)
    Lambda = 10

    print("------------------------------------------------")
    print("---> Test 2")
    print("theta:", theta)
    print("lambda:", Lambda)
    print("cost:", reg.cost(theta, X, y, Lambda), "\n(expected: 3.16)")
    print("grad [first 5]:", reg.gradient(theta, X, y, Lambda)[:5], 
                "\n(expected: [0.3460 0.1614 0.1948 0.2269 0.0922 ...])")

    # Gradien descent (with alpha=[1, 5, 10])
    num_iter = 500
    alphas = [0.1, 0.5, 1]
    Lambda = 1

    #Xn = reg.norm_m(X,list(range(1,n)), t='s')    # Xn = normalized X
    Xn = X  # En este caso, no normalizamos

    j_alpha = {}
    for alpha in alphas:
        theta = np.zeros(n)         # initial theta (zeros)
        theta_list, j_list = reg.gradient_descent(theta, Xn, y, num_iter, alpha, Lambda)
        j_alpha[alpha] = j_list

    # plot gradients
    plot_gradient(num_iter, j_alpha)

    print("------------------------------------------------")
    print("---> Gradient descent (", num_iter, "iterations)")
    theta_optim = np.array(theta_list[-1])
    cost_optim = j_list[-1]
    print("theta_optim:", theta_optim)
    print("cost_optim:", cost_optim)

    # using contour
    plot_decission_boundary(theta_optim, Xn, y, degree)

    # Compute accuracy pn the training set
    print("------------------------------------------------")
    print("---> Prediction")
    print("Compute accuracy on our training set...")
    predictions = (reg.hypothesis(theta_optim, Xn) >= 0.5) * 1  # vector of 1's and 0's
    accuracy = ((predictions == y).sum())*100.0/len(y)          # compare with y
    print("Accuracy on training set:",  f'{accuracy: .2f}')
    