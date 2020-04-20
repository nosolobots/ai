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

    #-------------------------------------------------------------------
    # Compute prediction on the training set

    print("------------------------------------------------")
    print("Compute prediction on training set:")

    p, pk = predict(Theta1, Theta2, X)

    # compare with y values
    accuracy = ((p == y).sum())*100.0/len(y)
    print("Accuracy on training set:",  f'{accuracy: .2f}')

    #-------------------------------------------------------------------
    # Test on random samples

    print("------------------------------------------------")
    print("Test on random samples:")

    test_random(5, Theta1, Theta2, X)

