import numpy as np

# Sum Of Difference
def SOD(Y, y):
    return np.sum(np.abs(Y - y))

# Mean Square Error
def MSE(Y, y):
    return ((Y - y)**2).mean(axis=0)