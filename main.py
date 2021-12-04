import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree
from parameters import Parameters
from sys import argv
from random import random
from fp import FP


np.seterr(all='ignore')

# f1
# X = np.random.uniform(-1, 1, 40).reshape(40, 1)
X = np.linspace(-1, 5, num=100).reshape(100, 1)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

Parameters.POP_SIZE         = 25
Parameters.MAX_EVAL         = 25000
Parameters.MAX_GEN          = 1000
Parameters.INIT_MIN_DEPTH   = 0
Parameters.INIT_MAX_DEPTH   = 6
Parameters.MAX_DEPTH        = 15
Parameters.VAR_NUM          = 1

# create variables list
VARIABLES = []
for i in range(Parameters.VAR_NUM):
    var = 'x' + str(i)
    VARIABLES.append(var)
Parameters.VARIABLES = VARIABLES

if len(argv) > 1:
    Parameters.POP_SIZE = int(argv[1])

fp = FP(pop_size=25,
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

fp.fit(X_train, y_train)
fp.export_best()

y_predict = fp.predict(X_test)
print(y_predict)

score = fp.score(y_test, y_predict)
print(score)

fp.plot(X_test, X_train, y_train, y_test, y_predict)

# x = Tree()
# x.create_tree('full', Parameters.INIT_MIN_DEPTH, Parameters.INIT_MAX_DEPTH)

# x.draw_tree('test', 'label')