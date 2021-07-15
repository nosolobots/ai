""" Use a 3-layer Neural Network to predict hand-written numbers.

    1st layer (input) - 400 features (29x20 pixel image) 
    2nd layer (hidden layer) - 25 neurons
    3rd layer (output) - 10 neurons (1 x number)

    ex3weights.mat - weigthed parameters of theta1 and theta2 transformation matrices file
"""
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from scipy.io import loadmat

import lib.logreg as reg
import lib.ann as ann

def predict(Theta1, Theta2, X):
    # number of samples
    m = X.shape[0]

    # Compute A1 (layer-1 activation)
    A1 = reg.hypothesis(Theta1.T, np.c_[np.ones((m,1)), X])

    # print("A1: ", A1.shape)

    # Compute A2 (output layer activation)
    A2 = reg.hypothesis(Theta2.T, np.c_[np.ones((m,1)), A1])

    # print("A2 (Z): ", A2.shape)    

    # Prediction:
    
    # matrix (m x labels) of 1's and 0's. There's a 1 on the predicted class (order: class-1, class-2,... , class-0)
    predict_k = (A2 >= 0.5) * 1
        
    # vector (m, 1) with the class number
    predict = np.argmax(predict_k, axis=1) + 1  

    return predict, predict_k

def test_random(digits, Theta1, Theta2, X):
    m = X.shape[0]  # samples
    n = X.shape[1]  # features

    computed_val = ""
    
    fig, axis = plt.subplots(nrows=1, ncols=digits, figsize=(8,8))
    
    for i in range(digits):
        # sample data
        Xs = X[np.random.randint(0, m+1),:].reshape(1, n)

        p_val = predict(Theta1, Theta2, Xs)[0][0]
        
        if p_val == 10: p_val = 0   # change class-10 (number 0)
        computed_val += str(p_val)

        axis[i].imshow(Xs.reshape(20, 20, order="F"), cmap = "gray_r")
        axis[i].axis("off")

    # prediction
    print("Predicted value: --->", computed_val)   

    plt.show()    

def check_backprop(Lambda=0.0):
    input_layer = 3
    hidden_layer = 5
    output_layer = 3
    m = 5

    nn = ann.NN(input_layer, hidden_layer, output_layer)

    y = 1 + np.arange(m)%output_layer
    
    X = np.zeros((m, input_layer))
    X = np.sin(np.arange(X.size)).reshape(X.shape)/10

    T1 = np.zeros((hidden_layer, input_layer + 1))
    T1 = np.sin(np.arange(T1.size)).reshape(T1.shape)/10

    T2 = np.zeros((output_layer, hidden_layer + 1))
    T2 = np.sin(np.arange(T2.size)).reshape(T2.shape)/10

    nn.set_parameters(T1, T2)

    # check without regularization
    J, grad = nn.cost_function(X, y, Lambda)

    # compute numerical for T1
    eps = 1e-4
    compT = np.zeros(T1.shape)
    for i in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            T_eps = np.array(T1)
            T_eps[i,j] -= eps
            nn.set_parameters(T_eps, T2)
            loss1 = nn.cost_function(X, y, Lambda)[0]
            T_eps[i,j] += 2*eps
            nn.set_parameters(T_eps, T2)
            loss2 = nn.cost_function(X, y, Lambda)[0]
            compT[i,j] = (loss2 - loss1)/(2*eps)

    res = np.c_[grad[0].reshape((grad[0].size,1)), compT.reshape((compT.size,1))]
    print("T1 check:\n", res)

    # compute numerical for T2
    eps = 1e-4
    compT = np.zeros(T2.shape)
    for i in range(T2.shape[0]):
        for j in range(T2.shape[1]):
            T_eps = np.array(T2)
            T_eps[i,j] -= eps
            nn.set_parameters(T1, T_eps)
            loss1 = nn.cost_function(X, y, Lambda)[0]
            T_eps[i,j] += 2*eps
            nn.set_parameters(T1, T_eps)
            loss2 = nn.cost_function(X, y, Lambda)[0]
            compT[i,j] = (loss2 - loss1)/(2*eps)

    res = np.c_[grad[1].reshape((grad[1].size,1)), compT.reshape((compT.size,1))]
    print("T2 check:\n", res)

if __name__ == "__main__":
    #-------------------------------------------------------------------
    # Load Data

    print("------------------------------------------------")
    print("Loading data...")

    # use loadmar to load matlab (octave) files
    print("Training set:")

    mat = loadmat('data/ex3data1.mat')

    # data is loaded as a dictionary    
    X = mat['X']
    y = np.array(mat['y'][:,0])  # create a vector from a (m,1) matrix

    print("X: ", X.shape)
    print("y: ", y.shape)

    # load parameters
    print("Neural network parameters:")

    mat = loadmat('data/ex3weights.mat')

    # data is loaded as a dictionary    
    Theta1 = mat['Theta1']
    Theta2 = mat['Theta2']

    print("Theta-1: ", Theta1.shape)
    print("Theta-2: ", Theta2.shape)

    nn = ann.NN(400, 25, 10)

    # Test sigmoid_gradient
    print("- TEST ---------------------------------------------")
    print(">> Sigmoid Gradient ")
    g = nn.sigmoid_gradient(np.array([-1, 0.5, 0, 0.5, 1]))
    print("sigmoid_gradient([-1, 0.5, 0, 0.5, 1]):", g)
    
    # Test cost function
    print("- TEST ---------------------------------------------")
    print(">> Cost Function ")
    nn.set_parameters(Theta1, Theta2)
    print("cost:", nn.cost_function(X, y)[0])
    print("expected cost: 0.287629")
    
    # Test cost function with regularization
    print("- TEST ---------------------------------------------")
    print(">> Cost Function w/ Regularization (lambda=1)")
    nn.set_parameters(Theta1, Theta2)
    print("cost:", nn.cost_function(X, y, Lambda=1)[0])
    print("expected cost: 0.383770")
    
    # Checking backpropagation
    print("- TEST ---------------------------------------------")
    print(">> Back Propagation Gradient vs Numerical Gradient")
    check_backprop()
    
    # Checking backpropagation with regularization Lamda=3.0
    print("- TEST ---------------------------------------------")
    print(">> Back Propagation Gradient vs Numerical Gradient w/ Regularization")
    check_backprop(3.0)
    
    # Training
    print("- TEST ---------------------------------------------")
    print(">> Training\n")
    
    num_iterations = 50 # 1500
    Lambda = 1.0
    alpha = 1.0 # 1.5

    nn.training(X, y, num_iterations, Lambda, alpha, plot=True)

    #-------------------------------------------------------------------
    # Test on random samples

    print("------------------------------------------------")
    print("Test on random samples:")

    test_random(10, nn.Theta1, nn.Theta2, X)
    
