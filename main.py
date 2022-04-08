import numpy as np
from sklearn.model_selection import train_test_split
from parameters import Parameters
from methods import Methods
from functions import *
from errors import *
from random import random
from fp import FP
from dfp import DFP



np.seterr(all='ignore')


# box series
# box_dataset = np.loadtxt('box.dat')
# X = box_dataset[:, [0, 1]]
# y = box_dataset[:, 2]
# t = np.arange(0, 290)


point_num = 20

# f1
# X = np.linspace(-1, 1, num=point_num).reshape(point_num, 1)
# y = X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

# f2
# X = np.linspace(-1, 1, num=point_num).reshape(point_num, 1)
# y = X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

# f3
# X = np.linspace(0, 1, num=point_num).reshape(point_num, 1)
# y = np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2)

# f4
# X = np.linspace(0, np.pi/2, num=point_num).reshape(point_num, 1)
# y = np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1

# f5
# X = np.linspace(0, 2, num=point_num).reshape(point_num, 1)
# y = np.log(X[:, 0] + 1) + np.log((X[:, 0]**2) + 1)

# f6
# X = np.linspace(0, 4, num=point_num).reshape(point_num, 1)
# y = np.sqrt(X[:, 0])

# f7
# X = np.linspace((0, 0), (1, 1), num=point_num).reshape(point_num, 2)
# y = np.sin(X[:, 0]) + np.sin(X[:, 1] ** 2)

# f8
X = np.linspace((0, 0), (1, 1), num=point_num).reshape(point_num, 2)
y = 2 * np.sin(X[:, 0]) * np.cos(X[:, 1])

# Take a dataset split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

# X_train = X[0:200, :]
# X_test = X[200:, :]
# y_train = y[0:200]
# y_test = y[200:]
# t_train = t[:200]
# t_test = t[200:]



Parameters.CONSTANTS = [1, 2]
Parameters.FEATURES = X.shape[1]
Parameters.CONSTANTS_TYPE = 'integer'
# Parameters.CONSTANTS_TYPE = 'range'
Parameters.Export_EXT = 'pdf'

# expressions = [Add, Sub, Mul, Div, Sin, Cos, Rlog, Exp, Pow]
expressions = [Add, Sub, Mul, Div, Sin, Cos, Rlog, Exp]
# expressions = [Add, Sub, Mul, Div, Sin, Cos]
# expressions = [Add, Sub, Mul, Div, Pow]
terminals = [Variable, Constant]

model = FP(pop_size=50,
        alpha=0.1,
        beta=0.5,
        gamma=1.0,
        max_evaluations=25000,
        initial_min_depth=0,
        initial_max_depth=6,
        max_depth=15,
        error_function=SOD,
        expressions=expressions,
        terminals=terminals,
        target_error=1e-5,
        verbose=True
        )

y_fitted = model.fit(X, y)
# y_fitted = fp.fit(X_train, y_train)
# y_pred = fp.predict(X_test)
# print(fp.best_individual.equation())
# print(fp.score(X_test, y_test))

Methods.plot(x_axis_train=X[:, 0],
                y_axis_train= y,
                y_axis_fitted=y_fitted)
# Methods.plot(x_axis_train=X[:, 0],
#                 y_axis_train= y,
#                 y_axis_fitted=y_fitted)

# Methods.plot(x_axis_train=t_train,
#             y_axis_train=y_train,
#             y_axis_fitted=y_fitted,
#             x_axis_test=t_test,
#             y_axis_test=y_test,
#             y_axis_pred=y_pred,
#             test_set=True)

# Methods.plot(x_axis_train=, y_axis_train, y_axis_fitted)