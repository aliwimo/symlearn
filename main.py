import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree
from parameters import Parameters
from methods import Methods
from errors import *
from sys import argv
from random import random
# from box import BOX
from fp import FP
from dfp import DFP


np.seterr(all='ignore')


# box series
box_dataset = np.loadtxt('box.dat')
X = box_dataset[:, [0, 1]]
y = box_dataset[:, 2]
t = np.arange(0, 290)


# f1
# X = np.linspace(-1, 1, num=20).reshape(20, 1)
# y = X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

# f2
# X = np.linspace(-1, 9, num=100).reshape(100, 1)
# y = X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

# f3
# X = np.linspace(0, 10, num=100).reshape(100, 1)
# y = np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2)

# f4
# X = np.linspace(-1, 9, num=100).reshape(100, 1)
# y = np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1

# f5
# X = np.linspace(0, 10, num=100).reshape(100, 1)
# y = np.log(X[:, 0] + 1) * np.cos((X[:, 0]**2) + 1)

# f6
# X = np.linspace(0, 20, num=100).reshape(100, 1)
# y = np.sqrt(X[:, 0])

# f7
# X = np.linspace((0, 0), (10, 10), 100).reshape(100, 2)
# y = np.sin(X[:, 0]) + np.sin(X[:, 1] ** 2)

# Take a dataset split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
# X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, shuffle=False)

X_train = X[0:200, :]
X_test = X[200:, :]
y_train = y[0:200]
y_test = y[200:]
t_train = t[:200]
t_test = t[200:]


# create variables list
VARIABLES = []
for i in range(X.shape[1]):
    var = 'x' + str(i)
    VARIABLES.append(var)
Parameters.VARIABLES = VARIABLES
Parameters.OPERATORS = ['+', '-', '*', '/']
Parameters.FUNCTIONS = ['sin', 'cos', 'exp', 'rlog']
Parameters.CONSTANTS = [-1, 1]
Parameters.CONSTANTS_TYPE = 'range'
Parameters.ERROR_FUNCTION = MSE


bx = DFP(
    pop_size=100,
    alpha=0.1,
    beta=0.5,
    gamma=1,
    max_evaluations=25000,
    initial_min_depth=0,
    initial_max_depth=6,
    max_depth=6,
    target_error=1e-6,
    verbose=True
)

y_fitted = bx.fit(X_train, y_train)
y_pred = bx.predict(X_test)
bx.export_best()

Methods.plot(x_axis_train=t_train,
            y_axis_train=y_train,
            y_axis_fitted=y_fitted,
            x_axis_test=t_test,
            y_axis_test=y_test,
            y_axis_pred=y_pred,
            test_set=True)

# Methods.plot(x_axis_train=X_train,
#             y_axis_train=y_train,
#             y_axis_fitted=y_fitted,
#             x_axis_test=X_test,
#             y_axis_test=y_test,
#             y_axis_pred=y_pred)

# Methods.plot(x_axis_train=X_train,
#             y_axis_train=y_train,
#             y_axis_fitted=y_fitted)

# bx.plot(X_train, X_test, y_train, y_test, y_predict)

# from sklearn.metrics import confusion_matrix
# score = confusion_matrix(y_test, y_predict)
# print(score)