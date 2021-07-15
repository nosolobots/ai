"""Handwritten Numbers OCR.
"""
import lib.ann as ann
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import sys

_DEFAULT_INPUT_LAYER = 400
_DEFAULT_HIDDEN_LAYER = 25
_DEFAULT_OUTPUT_LAYER = 10

_DEFAULT_DATA_FILE = "data/numbers_data.mat"
_DEFAULT_PARAMS_NPZ_FILE = "data/params.npz"
_DEFAULT_PARAMS_MAT_FILE = "data/params.mat"

_DEFAULT_LAMBDA = 1.0
_DEFAULT_ALPHA = 1.0
_DEFAULT_NUM_ITER = 50

def opc_create_ANN():
    clearscreen()

    print(">> Create a new _ANN_...\n")

    input_layer = _DEFAULT_INPUT_LAYER
    cin = input("Input layer units [" + str(_DEFAULT_INPUT_LAYER) + "]: ") 
    if(len(cin) != 0): input_layer = int(cin)

    hidden_layer = _DEFAULT_HIDDEN_LAYER
    cin = input("Hidden layer units [" + str(_DEFAULT_HIDDEN_LAYER) + "]: ") 
    if(len(cin) != 0): hidden_layer = int(cin)

    output_layer = _DEFAULT_OUTPUT_LAYER
    cin = input("Output layer units [" + str(_DEFAULT_OUTPUT_LAYER) + "]: ") 
    if(len(cin) != 0): output_layer = int(cin)

    nn = ann.NN(input_layer, hidden_layer, output_layer)

    print("Created.")

    return nn    

def opc_load_train_test_data():
    clearscreen()

    print(">> Load training and test data...\n")

    filename = _DEFAULT_DATA_FILE
    cin = input("filename [" + filename + "]: ") 
    if(len(cin) != 0): filename = cin

    mat = loadmat(filename)    

    # .mat data is loaded as a dictionary    
    X = mat['X']            # (m,n)
    Y = np.array(mat['y'])  # (m,1)

    m = X.shape[0]          # no. of data samples

    # shuffle data
    index = np.arange(m)
    np.random.shuffle(index)

    # create training (80%) and test (20%) sets
    p = int(m*0.8)

    X_train = X[index][:p,:]
    y_train = np.array(Y[index][:p,0])

    print("Training data set created.")
    show = input("Show samples [y/N]? ").upper()
    if show == "Y": plot_samples(X_train)

    X_test = X[index][p:,:]
    y_test = np.array(Y[index][p:,0])

    print("Testing data set created.")
    show = input("Show samples [y/N]? ").upper()
    if show == "Y": plot_samples(X_train)

    return X_train, y_train, X_test, y_test

def opc_load_mat_params(nn):
    clearscreen()

    print(">> Load .mat params...\n")

    filename = _DEFAULT_PARAMS_MAT_FILE
    cin = input("filename [" + filename + "]: ") 
    if(len(cin) != 0): filename = cin

    print("loading...")
    mat = loadmat(filename)

    # data is loaded as a dictionary    
    Theta1 = mat['Theta1']
    Theta2 = mat['Theta2']    

    print("setting params...")
    nn.set_parameters(Theta1, Theta2)

    print("Loaded.")

def opc_load_npz_params(nn):
    clearscreen()

    print(">> Load .npz params...\n")

    filename = _DEFAULT_PARAMS_NPZ_FILE
    cin = input("filename [" + filename + "]: ") 
    if(len(cin) != 0): filename = cin

    print("loading...")
    dat = np.load(filename)

    # data is loaded as a dictionary    
    Theta1 = dat[dat.files[0]]
    Theta2 = dat[dat.files[1]]    

    print("setting params...")
    nn.set_parameters(Theta1, Theta2)

    print("Loaded.")

def opc_train(nn, X, y):
    clearscreen()

    print(">> Training...\n")
    
    num_iterations = 50
    cin = input("No. iterations [50]: ")
    if(len(cin) != 0): num_iterations = int(cin)
    
    alpha = 1.0 # 1.5
    cin = input("Alpha (learning rate) [1.0]: ")
    if(len(cin) != 0): alpha = float(cin)
    
    Lambda = 1.0
    cin = input("Lambda (regularization param) [1.0]: ")
    if(len(cin) != 0): Lambda = float(cin)

    plot = True
    cin = input("Plot gradient [Y/n]: ").upper()
    if(cin=="N"): plot = False

    nn.training(X, y, num_iterations, Lambda, alpha, plot=plot)

def opc_accuracy(nn, X, y):
    clearscreen()

    # accuracy on the training set
    print("--> Compute accuracy on the data set...\n")

    p, pk = nn.predict(X)

    accuracy = ((p == y).sum())*100.0/len(y)
    print("Accuracy:",  f'{accuracy: .2f}')  

def opc_save_npz_params(nn):
    clearscreen()

    print(">> Save .npz params...\n")

    filename = _DEFAULT_PARAMS_NPZ_FILE
    cin = input("filename [" + filename + "]: ") 
    if(len(cin) != 0): filename = cin

    print("saving...")
    np.savez(filename, nn.Theta1, nn.Theta2)

    print("Saved.")

def opc_plot_layer_units(nn, width=0):
    clearscreen()

    print(">> Plotting layer units...\n")

    plot = True
    cin = input("Plot hidden layer units [Y/n]: ").upper()
    if(cin=="N"): plot = False

    if plot:
        plot_hidden_layer(nn)

    plot = False
    cin = input("Plot output layer units [y/N]: ").upper()
    if(cin=="Y"): plot = True    

    if plot:
        plot_output_layer(nn)    

def opc_test_ANN(nn, X, y):
    clearscreen()

    print(">> Testing on random data from testing set...\n")

    digits = 5
    cin = input("How many digits [5]? ")
    if len(cin)!=0: digits = int(cin)

    m = X.shape[0]  # samples
    n = X.shape[1]  # features

    computed_val = ""

    fig, axis = plt.subplots(nrows=1, ncols=digits, figsize=(8,3))
    
    for i in range(digits):
        # sample data
        Xs = X[np.random.randint(0,m),:].reshape(1, n)

        p_val = nn.predict(Xs)[0][0]
        
        if p_val == 10: p_val = 0   # change class-10 (number 0)
        computed_val += str(p_val)

        axis[i].imshow(Xs.reshape(20, 20, order="F"), cmap = "gray_r")
        axis[i].axis("off")

    # prediction
    print("Predicted value: --->", computed_val)   

    plt.show()     

def plot_samples(X):
    # plot some samples of the loaded data
    fig, axis = plt.subplots(10, 10, figsize=(8,4))

    # randomly extract 100 samples, plotting them in a 10x10 composition
    # each sample has 400 pixels (features) that we have to reshape in
    # a 20x20 image (order=F forces upright orientation)
    for i in range(10):
        for j in range(10):
            axis[i,j].imshow(
                X[np.random.randint(0, X.shape[0]+1),:].reshape(20, 20, order="F"),
                cmap = "gray_r")
            axis[i,j].axis("off")

    plt.show()

def plot_hidden_layer(nn, width=0):
    clearscreen()

    print(">> Plotting hidden layer units...\n")

    Theta = nn.Theta1[:,1:] # skip 1's column

    example_width = width
    if example_width==0:
        example_width = int(round(np.sqrt(Theta.shape[1])))

    # compute rows, cols
    m,n = Theta.shape
    example_height = int(n/example_width)

    # compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # padding between image
    pad = 1

    # plot some samples of the loaded data
    fig, axis = plt.subplots(display_rows, display_cols, 
                                figsize=(8, 8))

    unit = 0
    for i in range(display_rows):
        for j in range(display_cols):
            axis[i,j].imshow(
                Theta[unit,:].reshape(example_width, example_height, order="F"),
                cmap = "gray")
            axis[i,j].axis("off")
            unit += 1

    plt.show()  

def plot_output_layer(nn, width=0):
    clearscreen()

    print(">> Plotting output layer units...\n")

    Theta = nn.Theta2[:,1:] # skip 1's column

    example_width = width
    if example_width==0:
        example_width = int(round(np.sqrt(Theta.shape[1])))

    # compute rows, cols
    m,n = Theta.shape
    example_height = int(n/example_width)

    # compute number of items to display
    display_rows = m
    display_cols = 1

    # padding between image
    pad = 1

    # plot some samples of the loaded data
    fig, axis = plt.subplots(display_rows, display_cols, 
                                figsize=(3, 8))

    unit = 0
    for i in range(display_rows):
        axis[i].imshow(
            Theta[unit,:].reshape(example_width, example_height, order="F"),
            cmap = "gray")
        axis[i].axis("off")
        unit += 1

    plt.show()      

def clearscreen():
    print("\033c")

def menu():
    opc = " "
    while(opc not in ("123456789TX")):
        clearscreen()

        print("\t###################################")
        print("\t#     HANDWRITTEN NUMBERS OCR     #")
        print("\t#---------------------------------#")
        print("\t#  1 - Create _ANN_               #")        
        print("\t#  2 - Load train/test (mat)      #")
        print("\t#  3 - Load parameters (mat)      #")
        print("\t#  4 - Load parameters (npz)      #")
        print("\t#  5 - Train _ANN_                #")
        print("\t#  6 - Accuracy on training data  #")
        print("\t#  7 - Accuracy on test data      #")
        print("\t#  8 - Save parameters (npz)      #")
        print("\t#  9 - Plot layer units           #")
        print("\t#  T - Test _ANN_                 #")
        print("\t#  X - Exit                       #")
        print("\t###################################")
        opc = input("\n\t> option: ").upper()

    return opc

if __name__ == "__main__":
    # _ANN_ Artificial Neural Network
    nn = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    opc = " "
    while(opc != "X"):
        opc = menu()
        
        if opc == "1":
            nn  = opc_create_ANN()

        elif opc == "2":
            X_train, y_train, X_test, y_test = opc_load_train_test_data()            

        elif opc == "3":
            if nn: 
                opc_load_mat_params(nn)
            else: 
                clearscreen()
                print("Create ANN first!")

        elif opc == "4":
            if nn: 
                opc_load_npz_params(nn)   
            else: 
                clearscreen()
                print("Create ANN first!")

        elif opc == "5":
            if nn and not X_train is None and not y_train is None: 
                opc_train(nn, X_train, y_train)
            else: 
                clearscreen()
                print("Create ANN and load data first!")

        elif opc == "6":
            if nn and not X_train is None and not y_train is None: 
                opc_accuracy(nn, X_train, y_train)
            else: 
                clearscreen()
                print("Create ANN and load data first!")
        
        elif opc == "7":
            if nn and not X_test is None and not y_test is None: 
                opc_accuracy(nn, X_test, y_test)
            else: 
                clearscreen()
                print("Create ANN and load data first!")

        elif opc == "8":
            if nn: 
                opc_save_npz_params(nn)
            else: 
                clearscreen()
                print("Create ANN first!")

        elif opc == "9":
            if nn: 
                opc_plot_layer_units(nn)
            else: 
                clearscreen()
                print("Create ANN first!")

        elif opc == "T":
            if nn and not X_test is None and not y_test is None: 
                opc_test_ANN(nn, X_test, y_test)
            else: 
                clearscreen()
                print("Create ANN and load data first!")
        else:
            continue

        input("\nPress [RET] to continue.")

    clearscreen()