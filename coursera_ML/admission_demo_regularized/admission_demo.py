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

    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')

    if show:
        plt.legend(['Not admitted', 'Admitted'])
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
    x1 = np.linspace(0, 100, 50)
    x2 = np.linspace(0, 100, 50)
    
    x1v, x2v = np.meshgrid(x1, x2)

    Z = np.zeros((len(x2), len(x1))) 
    for i in range(len(x2)):
        for j in range(len(x1)):
            Z[i,j] = predict(x1[j], x2[i], theta, degree)

    X = reg.map_features(x1, x2, degree)
    Xn = reg.norm_m(X,list(range(1, X.shape(1))), t='s') 

    Z = Xn@theta

    for i in range(len(x2)):
        for j in range(len(x1)):
            Z[i,j] = mapFeaturePlot(x1[j], x2[i], degree)@theta
            #Z[i,j] = mapFeaturePlot(x1[j], x2[i], degree)@np.array([-1, 0, 0, 1, 0, 4])

    # filtramos próximos a 0
    F = np.where(reg.sigmoid(Z)>=0.5)
    
    plt.plot(x1[F[1]], x2[F[0]], 'k+')
    #cmap = plt.contourf(x1, x2, Z, levels=10)  # contour color map
    #plt.contour(x1, x2, Z, 0, colors='k')

    # plot data
    plot_data(X, y, show=False)
    """
    neg = np.where(y==0)    # list of row-indices where y=0
    pos = np.where(y==1)    # list of row-indices where y=1
    plt.scatter(X[neg][:,1],X[neg][:,2],c="b",marker="x",label="QA not passed")
    plt.scatter(X[pos][:,1],X[pos][:,2],c="r",marker="+",label="QA Passed")
    """
    #plt.legend(['Decision boundary', 'QA not passed', 'QA Passed'])
    plt.title('Logistic Regression (contour)')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    #plt.colorbar(cmap)
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

def predict(theta, x1, x2, mean, std, degree):
    # feature matrix
    x_test = reg.map_features(np.array([x1]), np.array([x2]), degree)[0] # first row

    # normalization with mean and std extracted from training set
    x_test[1:] = (x_test[1:] - mean)/std
    return reg.hypothesis(theta_optim, x_test)    

if __name__ == "__main__":
    # load data
    df = pd.read_csv('data/ex2data1.txt', header=None)
    df.columns = ['EX1', 'EX2', 'ADM']

    # number of samples
    m = df.shape[0]

    # create X matrix
    X = df.drop(columns=['ADM']).values
    X = np.c_[np.ones([m, 1]), X]

    # create y results
    y = df.ADM.values

    # plot input data
    plot_data(X, y)

    # create X with new polynomical features from X1, X2
    degree = 6
    X = reg.map_features(df.EX1.values, df.EX2.values, degree)

    # numer of fetaures (including X0)
    n = X.shape[1]

    # Gradien descent (with alpha=[1, 5, 10])
    num_iter = 500
    alphas = [1]
    Lambda = 1

    Xn, mean, std = reg.norm_m(X, list(range(1,n)), t='s')    # Xn = normalized X

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

    print("------------------------------------------------")
    print("---> Prediction")
    print("Compute accuracy on our training set...")

    predictions = (reg.hypothesis(theta_optim, Xn) >= 0.5) * 1  # vector of 1's and 0's
    accuracy = ((predictions == y).sum())*100.0/len(y)          # compare with y

    print("Accuracy on training set:",  f'{accuracy: .2f}')

    print("------------------------------------------------")
    print("---> Prediction")
    print("For a student with EX1=70, EX2=50...")
    
    # notas
    x1 = 70
    x2 = 50
   
    admission_probability = 100*predict(theta_optim, x1, x2, mean, std, degree)
    print("Probability of admission (%):", f'{admission_probability:.2f}')
    
    # PLOT DECISSION BOUNDARY 1
    #plot_decission_boundary(theta_optim, Xn, y, degree)    

    # PLOT DECISSION BOUNDARY 2
    x1v = np.linspace(30, 100, 70)
    x2v = np.linspace(30, 100, 70)

    """
    Z = np.zeros((len(x1v)*len(x2v),3))

    i = 0
    for x1 in x1v:
        for x2 in x2v:
            Z[i,0] = x1
            Z[i,1] = x2
            Z[i,2] = predict(theta_optim, x1, x2, mean, std, degree)
            i += 1
    """
    X1, X2 = np.meshgrid(x1v, x2v)
    
    l1 = len(x1v)
    l2 = len(x2v)
    
    Z = np.c_[X1.reshape(l1*l2,1), X2.reshape(l1*l2,1), np.zeros(l1*l2)]
    
    for m in Z:
        m[2] = predict(theta_optim, m[0], m[1], mean, std, degree)

    # filter where prediction is 50% aprox (boundary)
    b = np.where(np.abs(Z[:,2] - 0.5) < 0.05)
    plt.plot(Z[b][:,0], Z[b][:,1], 'gd', markersize=5)
    
    # plot data
    plot_data(X, y, show=False)

    plt.show()

    # PLOT DECISSION BOUNDARY 3
    h = Z[:,2].reshape(l2, l1)
    
    cmap = plt.contourf(x1v, x2v, h, levels = 8)
    #cmap = plt.contour(x1v, x2v, h, levels = 8)
    plt.colorbar(cmap)

    #plt.plot(Z[b][:,0], Z[b][:,1], 'k.', markersize=10)

    # plot data
    plot_data(X, y, show=False)

    plt.show()

    #cmap = plt.contourf(x1, x2, Z, levels=10)  # contour color map
    #plt.contour(x1v, x2v, Z)

    """
    neg = np.where(y==0)    # list of row-indices where y=0
    pos = np.where(y==1)    # list of row-indices where y=1
    plt.scatter(X[neg][:,1],X[neg][:,2],c="b",marker="x",label="QA not passed")
    plt.scatter(X[pos][:,1],X[pos][:,2],c="r",marker="+",label="QA Passed")
    """


