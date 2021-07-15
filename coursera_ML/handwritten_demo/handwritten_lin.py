import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy.io import loadmat

import lib.logreg as reg

def plot_samples(X):
    # plot some samples of the loaded data
    fig, axis = plt.subplots(10, 10, figsize=(8,8))

    # randomly extract 100 samples, plotting them in a 10x10 composition
    # each sample has 400 pixels (features) that we have to reshape in
    # a 20x20 image (order=F forces upright orientation)
    for i in range(10):
        for j in range(10):
            axis[i,j].imshow(
                X[np.random.randint(0, 5001),:].reshape(20, 20, order="F"),
                cmap = "hot")
            axis[i,j].axis("off")

    plt.show()

def plot_gradient(num_iter, j_dict):
    x = np.arange(num_iter)

    for k, j_list in j_dict.items():
        plt.plot(x, j_list)

    plt.xlabel('No. iterations')
    plt.ylabel('Cost (J)')
    plt.title('Minimize cost function J\n(gradient descent)')

    classes = list(j_dict.keys())
    plt.legend(['class: ' + str(k) for k in classes])

    plt.show()

if __name__ == "__main__":
    # load data
    # use loadmar to load matlab (octave) files
    mat = loadmat('data/ex3data1.mat')

    # data is loaded as a dictionary    
    X = mat["X"]
    y = np.array(mat["y"][:,0])  # create a vector from a (m,1) matrix

    print("X: ", X.shape)
    print("y: ", y.shape)

    # plot some samples 
    #plot_samples(X)

    # Test cost function and gradient
    print("------------------------------------------------")
    print(" Test cost function and gradient")
    theta_t = np.array([-2,-1,1,2])
    X_t = np.array([np.linspace(0.1,1.5,15)]).reshape(3,5).T
    X_t = np.hstack((np.ones((5,1)), X_t))
    y_t = np.array([1,0,1,0,1])
    J = reg.cost(theta_t, X_t, y_t, 3)
    grad = reg.gradient(theta_t, X_t, y_t, 3)
    print("Cost:",J,"\nExpected cost: 2.534819")
    print("Gradients:\n",grad,"\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

    # Compute gradient descent for each class
    print("------------------------------------------------")
    print(" Compute gradient descent for each class")
    
    m,n = X.shape
    num_labels = 10

    X = np.c_[np.ones((m,1)), X]   # add one's column

    theta_class = np.zeros([num_labels, n+1])

    num_iter = 100
    alpha = 1.5
    Lambda = 1    
    initial_theta = np.zeros(n+1)

    j_class = {}
    for k in range(1, num_labels+1):
        print("Class:", k)
        # np.where(y==k, 1, 0) builds a vector where value is 1 if y==k; 0 otherwise
        theta_list, j_list = reg.gradient_descent(initial_theta, X, np.where(y==k, 1, 0), num_iter, alpha, Lambda)
        j_class[k] = j_list
        theta_class[k-1,:] = np.array(theta_list[-1])  # order: class-1, class-2,... , class-0

    plot_gradient(num_iter, j_class)

    # Compute prediction on the training set
    print("------------------------------------------------")
    print(" Compute prediciton on training set")

    # matrix (m x labels) of 1's and 0's. There's a 1 on the predicted class (order: class-1, class-2,... , class-0)
    predict_k = (reg.hypothesis(theta_class.T, X) >= 0.5) * 1

    # vector (m, 1) with the class number
    predict = np.argmax(predict_k, axis=1) + 1  

    # compare with y values
    accuracy = ((predict == y).sum())*100.0/len(y)
    print("Accuracy on training set:",  f'{accuracy: .2f}')

    
    



