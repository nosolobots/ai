import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import lib.logreg as reg

def plot_data(X, y, show=True):
    # [x1, x2] where admission = 0
    neg = np.where(y==0)
    X_0 = X[neg][:, 1:]

    # [x1, x2] where admission = 1
    pos = np.where(y==1)
    X_1 = X[pos][:, 1:]

    plt.plot(X_0[:,0], X_0[:,1], 'rx')
    plt.plot(X_1[:,0], X_1[:,1], 'bo')

    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')

    if show:
        plt.legend(['Not admitted', 'Admitted'])
        plt.title('Input Data')
        plt.show()

def plot_decission_boundary(theta, X, y, show=True):
    # plot data
    plot_data(X, y, show=False)

    x = np.arange(X[:,1].min(), X[:,1].max())
    y = -(theta[0] + theta[1]*x)/theta[2]

    plt.plot(x, y, 'k')

    if show:
        plt.legend(['Not admitted', 'Admitted', 'Decision boundary'])
        plt.title('Logistic Regression')
        plt.show()

def plot_decission_boundary2(theta, X, y):
    # plot data
    plot_data(X, y, show=False)

    x1 = np.arange(X[:,1].min(), X[:,1].max())
    x2 = np.arange(X[:,2].min(), X[:,2].max())
    X, Y = np.meshgrid(x1, x2)
    Z = np.zeros(X.shape)

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i][j] = theta[0] + theta[1]*x1[i] + theta[2]*x2[j]

    plt.contour(X, Y, Z.T, 0)

    plt.legend(['Not admitted', 'Admitted', 'Decision boundary'])
    plt.title('Logistic Regression (contour)')
    plt.show()    

def plot_prediction(theta, X, y, sample, prob_1):
    # plot data and descission boundary
    plot_decission_boundary(theta, X, y, show=False)

    plt.plot(sample[1], sample[2], 'yo')

    legend = ['Not admitted', 'Admitted', 'Decision boundary', 'Prediction: ' + f'{prob_1:.2f}' + '%']
    plt.legend(legend)
    plt.title('Prediction (sample)')
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

def plot_input_data(X, y):
    # [x1, x2] where admission = 0
    neg = np.where(y==0)
    X_0 = X[neg][:, 1:]

    # [x1, x2] where admission = 1
    pos = np.where(y==1)
    X_1 = X[pos][:, 1:]

    plt.plot(X_0[:,0], X_0[:,1], 'rx')
    plt.plot(X_1[:,0], X_1[:,1], 'bo')

    plt.legend(['Not admitted', 'Admitted'])
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    plt.title('INPUT DATA')

    plt.show()    

if __name__ == "__main__":
    # load data
    df = pd.read_csv('data/ex2data1.txt', header=None)
    df.columns = ['EX1', 'EX2', 'ADM']

    # number of samples
    m = df.shape[0]

    # create X matrix
    X = df.drop(columns=['ADM']).values
    X = np.c_[np.ones([m, 1]), X]

    # numer of fetaures (including X0)
    n = X.shape[1]

    # create y results
    y = df.ADM.values

    # plot input data
    plot_data(X, y)

    # cost and gradient test 1
    # theta = [0, 0, 0]
    # expected cost = 0.693
    # expected gradient = [-0.10 -12.00 -11.26]
    theta = np.zeros(n)

    print("------------------------------------------------")
    print("---> Test 1")
    print("theta:", theta)
    print("cost:", reg.cost(theta, X, y), "(expected: 0.693)")
    print("grad:", reg.gradient(theta, X, y), "(expected: [-0.10 -12.00 -11.26])")

    # cost and gradient test 2
    # theta = [-24, 0.2, 0.2]
    # expected cost = 0.218
    # expected gradient = [0.043 2.566 2.647]
    theta = np.array([-24, 0.2, 0.2])

    print("------------------------------------------------")
    print("---> Test 2")
    print("theta:", theta)
    print("cost:", reg.cost(theta, X, y), "(expected: 0.218)")
    print("grad:", reg.gradient(theta, X, y), "(expected: [0.043 2.566 2.647])")

    # Gradien descent (with alpha=[1, 5, 10])
    num_iter = 500
    alphas = [0.01, .1, 1]

    Xn = reg.norm_m(X,[1,2], t='s')    # Xn = normalized X

    j_alpha = {}
    for alpha in alphas:
        theta = np.zeros(n)         # initial theta (zeros)
        theta_list, j_list = reg.gradient_descent(theta, Xn, y, num_iter, alpha)
        j_alpha[alpha] = j_list

    # plot gradients
    plot_gradient(num_iter, j_alpha)

    print("------------------------------------------------")
    print("---> Gradient descent (", num_iter, "iterations)")
    theta_optim = np.array(theta_list[-1])
    cost_optim = j_list[-1]
    print("theta_optim:", theta_optim)
    print("cost_optim:", cost_optim)

    print("------------------------------------------------")
    print("---> Plot decision boundary")
    plot_decission_boundary(theta_optim, Xn, y)
    
    # using contour
    plot_decission_boundary2(theta_optim, Xn, y)

    print("------------------------------------------------")
    print("---> Prediction")
    print("For a student with scores 45 85, I predict...")
    x_test = np.append(np.ones(1), reg.norm_v(np.array([45, 85])))
    print("Normalized vector:", x_test)
    admission_probability = 100*reg.hypothesis(theta_optim, x_test)
    print("Probability of admission (%):", admission_probability)
    plot_prediction(theta_optim, Xn, y, x_test, admission_probability)    

    print("------------------------------------------------")
    print("---> Prediction on training set")
    print("Compute accuracy on our training set...")
    predictions = (reg.hypothesis(theta_optim, Xn) >= 0.5) * 1  # vector of 1's and 0's
    accuracy = ((predictions == y).sum())*100.0/len(y)          # compare with y
    print("Accuracy on training set:",  f'{accuracy: .2f}')