"""Main module that is used to test package"""

# pylint: disable=unused-import
# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sklearn metrics and utilities
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import sklearn models
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import tree

# import core dependicies
from symlearn.core.parameters import Parameters
from symlearn.core.functions import *
from symlearn.core.errors import *

# import models
from symlearn.models import FFP, DFFP, IPP

# supress numpy warnings
np.seterr(all='ignore')

# create random data
X_train = np.random.uniform(0, 10, (75, 3))
y_train = np.random.uniform(0, 10, 75)
X_test = np.random.uniform(0, 10, (25, 3))
y_test = np.random.uniform(0, 10, 25)

# set global parameters
Parameters.CONSTANTS = [-5, 5]
Parameters.FEATURES = X_train.shape[1]
Parameters.CONSTANTS_TYPE = 'range'
expressions = [Add, Sub, Mul]
terminals = [Variable, Constant]


# choose a model to train
# model = MLPRegressor(max_iter=5000, hidden_layer_sizes=(4, 4, 4, ))
# model = tree.DecisionTreeRegressor()
# model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=2)
# model = make_pipeline(StandardScaler(), SVR(C=100.0, coef0=1.0, kernel='poly', max_iter=5000))

print("FFP")
model = FFP(pop_size=50,
        max_evaluations=2500,
        initial_min_depth=0,
        initial_max_depth=6,
        min_depth=1,
        max_depth=15,
        error_function=sum_of_difference,
        expressions=expressions,
        terminals=terminals,
        target_error=0,
        verbose=False
        )

# fit data into model
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# print results of the model
train_score = sum_of_difference(y_train, y_fit)
test_score = sum_of_difference(y_test, y_pred)
print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

print("\nDFFP")
model = DFFP(pop_size=50,
        alpha=0.01,
        beta=0.05,
        gamma=0.1,
        max_evaluations=5000,
        initial_min_depth=0,
        initial_max_depth=6,
        min_depth=1,
        max_depth=15,
        error_function=sum_of_difference,
        expressions=expressions,
        terminals=terminals,
        target_error=0,
        verbose=False
        )

# fit data into model
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# print results of the model
train_score = sum_of_difference(y_train, y_fit)
test_score = sum_of_difference(y_test, y_pred)
print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

print("\nIPP")
model = IPP(pop_size=100,
            donors_number=3,
            receivers_number=3,
            max_evaluations=2500,
            initial_min_depth=0,
            initial_max_depth=6,
            min_depth=1,
            max_depth=15,
            error_function=sum_of_difference,
            expressions=expressions,
            terminals=terminals,
            target_error=0,
            verbose=False
            )

# fit data into model
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# print results of the model
train_score = sum_of_difference(y_train, y_fit)
test_score = sum_of_difference(y_test, y_pred)
print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

# # plot model graph
# x_axis = range(100)
# plt.clf()
# ax = plt.axes()
# ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
# plt.ylabel("BTC Price ($)")
# line = [None, None, None, None]
# line[0], = ax.plot(x_axis[:75], y_train, linestyle=':',
#                    color='black', linewidth=0.7, zorder=2, label='Targeted')
# line[1], = ax.plot(x_axis[:75], y_fit, linestyle='-',
#                    color='red', linewidth=0.7, zorder=3, label='Trained')
# line[2], = ax.plot(x_axis[75:], y_test, linestyle=':',
#                    color='black', linewidth=0.7, zorder=2)
# line[3], = ax.plot(x_axis[75:], y_pred, linestyle='-',
#                    color='blue', linewidth=0.7, zorder=3, label='Predicted')
# plt.axvline(x=x_axis[0], linestyle='-', color='black', linewidth='1')
# fig = plt.gcf()
# fig.set_size_inches(13.66, 6.66)
# fig.savefig("temp/figure.png", dpi=100)
# plt.draw()
# plt.legend()
# plt.show()
