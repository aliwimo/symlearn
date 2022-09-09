import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Sum Of Difference
def SOD(Y, y):
    return np.sum(np.abs(Y - y))

# Mean Square Error
def MSE(Y, y):
    # return mean_squared_error(Y, y)
    return ((Y - y)**2).mean(axis=0)

# Root Mean Square Error
def RMSE(Y, y):
    return mean_squared_error(Y, y, squared=False)
    # return np.sqrt(((Y - y)**2).mean(axis=0))

# Mean Absolute Error
def MAE(Y, y):
    # return mean_absolute_error(Y, y)
    return np.sum(np.abs(Y - y)) / Y.shape[0]

# Inverse of R2 Score
def IR2(Y, y):
    return 1 - r2_score(Y, y)
    # Y_bar = Y.mean()
    # ss_tot = ((Y-Y_bar)**2).sum()
    # ss_res = ((Y-y)**2).sum()
    # return 1 - (ss_res/ss_tot)