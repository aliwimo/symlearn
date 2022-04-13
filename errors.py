import numpy as np

# Sum Of Difference
def SOD(Y, y):
    return np.sum(np.abs(Y - y))

# Mean Square Error
def MSE(Y, y):
    return ((Y - y)**2).mean(axis=0)

# Root Mean Square Error
def RMSE(Y, y):
    return np.sqrt(((Y - y)**2).mean(axis=0))

def MAE(Y, y):
    return np.sum(np.abs(Y - y)) / Y.shape[0]