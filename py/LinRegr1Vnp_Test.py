#!/usr/bin/env python3

"""Regresión Lineal para 1 variable.
   Utiliza numpy para la generación de valores.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt 

def initDataset(num_values=20, range=0, y_offset=0):
    """Inicializa las estructuras de datos.
    """
    # secuencia de valores en el eje X espaciados uniformemente
    data_x = np.arange(num_values)
    data_y = np.random.random(num_values)*range + data_x + y_offset

    # dataset
    return np.array((data_x, data_y))

def createGraphic(data, regression):
    """Muestra el gráfico
    """
    plt.plot(data[0], data[1], 'ro', regression[0], regression[1])
    plt.show()

def calculateRegression(dataset, numvars = 1):
    """Calcula la regresión lineal para el dataset y número de features suministrado.
    """
    # params
    alpha = 0.01
    epsilon = 1E-7
    alpham = alpha/len(dataset[0])
    
    # matrices
    TETHA = np.zeros(numvars + 1)
    TEMP = np.zeros(numvars + 1)

    X1 = np.array((np.ones(len(dataset[0])), dataset[0]))  # 1's column added to X
    Y = dataset[1]

    while(True):
        TEMP = TETHA - alpham*((X1.T.dot(TETHA) - Y).dot(X1.T))

        if np.abs((TETHA-TEMP).max()) < epsilon:
            TETHA = np.array(TEMP)
            break

        TETHA = np.array(TEMP)

    return [TETHA, np.array((dataset[0], X1.T.dot(TETHA)))]

def mce(dataset, tetha):
    X1 = np.array((np.ones(len(dataset[0])), dataset[0]))  # 1's column added to X
    Y = dataset[1]
    
    return 1/(2*len(dataset[0])) * (((X1.T.dot(tetha) - Y)**2).sum())

if __name__ == "__main__":
    dataset = None

    if(len(sys.argv) == 4):
        dataset = initDataset(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    else:
        print("uso:", os.path.basename(sys.argv[0]), "<num_values> <rango> <y_offset>")
        dataset = initDataset()

    # Regression lineal
    tetha, regression = calculateRegression(dataset)
    print("Tetha:", tetha)    
    print("MCE =", mce(dataset, tetha))

    # Mostramos el gráfico
    createGraphic(dataset, regression)