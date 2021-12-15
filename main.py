import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree
from parameters import Parameters
from methods import Methods
from sys import argv
from random import random
from fp import FP
from dfp import DFP
# from box import FP


np.seterr(all='ignore')


# box series
# box_dataset = np.loadtxt('box.dat')
# X = box_dataset[:, [0, 1]]
# y = box_dataset[:, 2]
# t = range(290)

# f1
X = np.random.uniform(-1, 1, 40).reshape(40, 1)
X = np.linspace(-1, 1, num=20).reshape(20, 1)
y = X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
# X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, shuffle=False)

# X_train = X[0:200, :]
# X_test = X[200:, :]
# y_train = y[0:200]
# y_test = y[200:]
# t_train = t[:200]
# t_test = t[200:]


# create variables list
VARIABLES = []
for i in range(X.shape[1]):
    var = 'x' + str(i)
    VARIABLES.append(var)
Parameters.VARIABLES = VARIABLES
Parameters.OPERATORS = ['+', '-', '*', '/']
Parameters.FUNCTIONS = ['sin', 'cos', 'exp', 'rlog']
Parameters.CONSTANTS = range(1, 2)

# fp = FP(pop_size=25,
#         alpha=0.1,
#         beta=0.5,
#         gamma=1.5,
#         max_evaluations=25000,
#         initial_min_depth=0,
#         initial_max_depth=6,
#         max_depth=15,
#         target_error=0.1,
#         verbose=True
#         )

# fp.fit(X_train, y_train)
# fp.export_best()
# y_predict = fp.predict(X_test)
# fp.plot(X_train, X_test, y_train, y_test, y_predict)

print('--------------------------------')

dfp = DFP(pop_size=25,
        alpha=0.1,
        beta=0.5,
        gamma=1.5,
        max_evaluations=25000,
        initial_min_depth=0,
        initial_max_depth=6,
        max_depth=15,
        target_error=0.1,
        verbose=True
        )

dfp.fit(X_train, y_train)
dfp.export_best()
y_predict = dfp.predict(X_test)
dfp.plot(X_train, X_test, y_train, y_test, y_predict)

