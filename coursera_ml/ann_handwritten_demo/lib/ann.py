"""ANN is an Artificial Neural Network of three layers (input, hidden, output).
"""

import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self, input_units, hidden_units, output_units):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.Theta1 = np.zeros([self.hidden_units, self.input_units + 1])
        self.Theta2 = np.zeros([self.output_units, self.hidden_units + 1])

    def set_random_parameters(self, eps=0.12):
        self.Theta1 = (np.random.random(self.Theta1.size)*2*eps - eps).reshape(self.Theta1.shape)
        self.Theta2 = (np.random.random(self.Theta2.size)*2*eps - eps).reshape(self.Theta2.shape)

    def set_parameters(self, Theta1_data, Theta2_data):
        if self.Theta1.shape != Theta1_data.shape or self.Theta2.shape != Theta2_data.shape:
            print("ERROR: set_parameters() incompatible shapes")
            return False

        self.Theta1 = np.array(Theta1_data)
        self.Theta2 = np.array(Theta2_data)
        
        return True

    def sigmoid(self, z):
        return 1/(1 + np.e**(-z))       

    def sigmoid_gradient(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))         

    def forward_propagation(self, X):
        # number of samples
        m = X.shape[0]
        
        # INPUT LAYER
        # add 1's column to X
        A1 = np.c_[np.ones((m, 1)), X]        

        # HIDDEN LAYER
        # hidden layer activations
        A2 = self.sigmoid(A1 @ self.Theta1.T)

        # OUTPUT LAYER
        # add a0=1 to hidden layer
        A2 = np.c_[np.ones((m, 1)), A2]        
        A3 = self.sigmoid(A2 @ self.Theta2.T)

        return A1, A2, A3

    def cost_function(self, X, y, Lambda=0.0):
        # number of samples
        m = X.shape[0]

        #-----------------------------------------------------------------------
        # FORWARD PROPAGATION

        # Output Layer activation
        A3 = self.forward_propagation(X)[2]

        #-----------------------------------------------------------------------
        # COST

        # Create an Y matrix of m-rows and k-columns
        # each row is a series of 0's with 1 in the corresponfig k
        Y = np.zeros((m, self.output_units))
        for j in range(m):
            Y[j, y[j]-1] = 1
        
        J = (1/m)*(-Y*np.log(A3) - (1 - Y)*np.log(1 - A3)).sum()    

        # Add regularization
        J = J + (Lambda/(2*m))*(np.sum(self.Theta1[:,1:]**2) + 
                                np.sum(self.Theta2[:,1:]**2))

        #-----------------------------------------------------------------------
        # BACKPROPAGATION
        D1 = np.zeros((self.Theta1.shape))
        D2 = np.zeros((self.Theta2.shape))

        for t in range(m):
            # sample-t
            x_t = X[t,:].reshape(1, self.input_units)
            
            # output-t
            y_t = Y[t,:].reshape(1, self.output_units)
            
            #--> forward propagation for x_t
            a1, a2, a3 = self.forward_propagation(x_t)

            #--> backpropagation for x_t

            # compute delta error for output layer
            d3 = a3 - y_t

            # compute delta error for hidden layer
            d2 = (d3 @ self.Theta2)*a2*(1 - a2)

            # compute gradient. Skip d2_0
            D2 += d3.T @ a2
            D1 += d2[0, 1:].reshape(self.hidden_units, 1) @ a1

        Theta1_grad = (1/m)*D1
        Theta2_grad = (1/m)*D2

        # Regularization
        Theta1_grad[:, 1:] += (Lambda/m)*self.Theta1[:,1:]
        Theta2_grad[:, 1:] += (Lambda/m)*self.Theta2[:,1:]

        return J, [Theta1_grad, Theta2_grad]

    def training(self, X, y, num_iterations, Lambda, alpha=1.0, verbose=True, plot=False):
        # initialize parameters
        self.set_random_parameters()

        # gradient descent
        J_list = []
        grad_List = []
        grad = [np.array(self.Theta1), np.array(self.Theta2)]
        for i in range(num_iterations):
            if verbose: print("Iteration",(i+1),"--> ",end="")
            
            j, grad = self.cost_function(X, y, Lambda)
            J_list.append(j)
            grad_List.append(grad)
            
            self.set_parameters(self.Theta1 - alpha*grad[0], self.Theta2 - alpha*grad[1])

            if verbose: print("cost:", J_list[-1])
    
        # plot gradient
        if plot:
            print("Plotting gradient descent...")
            plt.title("Gradient Descent")
            plt.plot(np.arange(num_iterations), J_list)
            plt.xlabel("num. iterations")
            plt.ylabel("cost (J)")
            plt.legend([])
            plt.show()

        # accuracy on the training set
        print("Compute prediction on training set:")

        p, pk = self.predict(X)

        accuracy = ((p == y).sum())*100.0/len(y)
        print("Accuracy on training set:",  f'{accuracy: .2f}')  

    def predict(self, X):
        #--> forward propagation
        # Output Layer activation
        A3 = self.forward_propagation(X)[2]

        # Prediction:
        
        # matrix (m x labels) of 1's and 0's. There's a 1 on the predicted class (order: class-1, class-2,... , class-0)
        predict_k = (A3 >= 0.5) * 1
            
        # vector (m, 1) with the class number
        predict = np.argmax(predict_k, axis=1) + 1  

        return predict, predict_k