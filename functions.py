import numpy as np

# add your custom functions here

def Add(X0, X1):
    return X0 + X1

def Sub(X0, X1):
    return X0 - X1

def Mul(X0, X1):
    return X0 * X1

def Div(X0, X1):
    sign_X1 = np.sign(X1)   # numpy.sign: X>=1 returns 1, X==0 returns 0, X<=1 returns -1 
    sign_X1[sign_X1 == 0] = 1
    return X0 / sign_X1

def Sin(X0):
    return np.sin(X0)

def Cos(X0):
    return np.cos(X0)

def Rlog(X0):
    return np.log(np.abs(X0) + 1e-6)

def Exp(X0):
    return np.exp(X0)