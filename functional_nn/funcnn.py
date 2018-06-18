import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x,turev=False):
    if (turev==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def veri():
        data = pd.read_csv('iris.csv', names=["x1", "x2", "x3", "x4", "y"])
        x = data.iloc[:, 0:4]
        y = data.iloc[:, 4:5].values
        print(x.shape)
        return x,y

def geri(y,w0,w1,w2,w3,l0,l1,l2,l3,l4,rate):
    l4_error = y - l4
    l4_delta = l4_error * sigmoid(l4, turev=True)

    l3_error = l4_delta.dot(w3.T)
    l3_delta = l3_error * sigmoid(l3, turev=True)

    l2_error = l3_delta.dot(w2.T)
    l2_delta = l2_error * sigmoid(l2, True)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, turev=True)

    w3 += l3.T.dot(l4_delta) / rate
    w2 += l2.T.dot(l3_delta) / rate
    w1 += l1.T.dot(l2_delta) / rate
    w0 += l0.T.dot(l1_delta) / rate

def nn(x,y,dongu,rate):
    cost = np.zeros(dongu)
    w0 = 2 * np.random.randn(4, 4) - 1
    w1 = 2 * np.random.randn(4, 4) - 1
    w2 = 2 * np.random.rand(4, 4) - 1
    w3 = 2 * np.random.rand(4, 1) - 1
    for i in range (dongu):
        l0 = x
        l1 = sigmoid(np.dot(l0, w0))
        l2 = sigmoid(np.dot(l1, w1))
        l3 = sigmoid(np.dot(l2, w2))
        l4 = sigmoid(np.dot(l3, w3))
        geri(y,w0,w1,w2,w3,l0,l1,l2,l3,l4,rate)
        error = l4 - y
        print(i, ".Epoch ", "Error:" + str(np.mean(np.abs(error))))
        cost[i] = np.mean(np.abs(error))
    error_graph(dongu,cost)

def error_graph(dongu,cost):
    fig, ax = plt.subplots()
    ax.plot(np.arange(dongu), cost, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


