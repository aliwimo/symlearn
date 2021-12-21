import numpy as np

def SUM_OF_DIFFERENCE(Y, y):
    return np.sum(np.abs(Y - y))

def MSE(Y, y):
    return ((Y - y)**2).mean(axis=0)