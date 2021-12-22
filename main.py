import numpy as np
from sklearn.model_selection import train_test_split
from tree import Tree
from parameters import Parameters
from methods_n import Methods
from errors import *
from sys import argv
from random import random
# from box import BOX
from fp import FP
import math
# from dfp import DFP

from functions_n import *


np.seterr(all='ignore')


# box series
box_dataset = np.loadtxt('box.dat')
X = box_dataset[:, [0, 1]]
y = box_dataset[:, 2]
t = np.arange(0, 290)


# point_num = 20

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
# X = np.linspace(0, math.pi/2, num=point_num).reshape(point_num, 1)
# y = np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1

# f5
# X = np.linspace(0, 2, num=point_num).reshape(point_num, 1)
# y = np.log(X[:, 0] + 1) * np.cos((X[:, 0]**2) + 1)

# f6
# X = np.linspace(0, 4, num=point_num).reshape(point_num, 1)
# y = np.sqrt(X[:, 0])

# f7
# X = np.linspace((0, 0), (1, 1), num=point_num).reshape(point_num, 2)
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
# VARIABLES = []
# for i in range(X.shape[1]):
#     var = 'x' + str(i)
#     VARIABLES.append(var)
# Parameters.VARIABLES = VARIABLES
# Parameters.OPERATORS = ['+', '-', '*', '/']
# Parameters.FUNCTIONS = ['sin', 'cos', 'exp', 'rlog']
Parameters.CONSTANTS = [-1, 1]
# Parameters.CONSTANTS_TYPE = 'intergers'
# Parameters.ERROR_FUNCTION = MSE
Parameters.FEATURES = X.shape[1]


# bx = DFP(
#     pop_size=100,
#     alpha=0.1,
#     beta=0.5,
#     gamma=1,
#     max_evaluations=25000,
#     initial_min_depth=0,
#     initial_max_depth=6,
#     max_depth=6,
#     target_error=1e-6,
#     verbose=True
# )

# y_fitted = bx.fit(X_train, y_train)
# y_pred = bx.predict(X_test)
# bx.export_best()

# Methods.plot(x_axis_train=t_train,
#             y_axis_train=y_train,
#             y_axis_fitted=y_fitted,
#             x_axis_test=t_test,
#             y_axis_test=y_test,
#             y_axis_pred=y_pred,
#             test_set=True)

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


# t = Tree()

# t.root.set_value('operator')

# l = Tree(parent_id=t.root.id)
# l.root.set_value('constant')

# r = Tree(parent_id=t.root.id)
# r.root.set_value('constant')

# t.left = l
# t.right = r

# print(t.tree_equation())
# t.draw_tree('tree', 'Label')

# defining parameters
expressions = [Add, Sub, Mul, Div, Sin, Cos, Rlog, Exp, Pow]
terminals = [Variable, Constant]

fp = FP(pop_size=50,
        alpha=0.1,
        beta=0.5,
        gamma=1.5,
        max_evaluations=100000,
        initial_min_depth=0,
        initial_max_depth=6,
        max_depth=17,
        error_function=MSE,
        expressions=expressions,
        terminals=terminals,
        target_error=0,
        verbose=True
        )

y_fitted = fp.fit(X_train, y_train)


# Parameters.FEATURES = 1
# X = np.linspace(-1, 1, num=4).reshape(4, 1)

# pop = Methods.generate_population(pop_size=10,
#                                 initial_min_depth=1,
#                                 initial_max_depth=3,
#                                 expressions=expressions,
#                                 terminals=terminals)

# x1 = Methods.generate_individual('grow', 1, 3, expressions, terminals)
# x2 = Methods.generate_individual('grow', 1, 3, expressions, terminals)

# Methods.export_graph(x1, 'x1_tree', 'x1')
# Methods.export_graph(x2, 'x2_tree', 'x2')

# x3 = Methods.share(x1, x2)
# Methods.export_graph(x3, 'x3_tree', 'x3')

        
# x1.insert_left_node(x3)
# x1.insert_right_node(x4)

# x5 = Variable()
# x6 = Constant()
# x2.insert_left_node(x5)
# x2.insert_right_node(x6)

# print(x0.get_equation())
# print(x0.get_sub_nodes())
# Methods.export_graph(x0, 'file_name', 'label')
# print(x0.output(X))


# for i in range(5):
#     x = Variable()
#     print(x)
