import numpy as np
import pandas as pd
from parameters import Parameters
from methods import Methods
from functions import *
from errors import *
from random import random
from fp import FP
from dfp import DFP

# example of calculate the mean squared error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.seterr(all='ignore')

df = pd.read_csv('btc_usd.csv')

# box series
box_dataset = np.loadtxt('box.dat')
X = box_dataset[:, [0, 1]]
y = box_dataset[:, 2]
t = np.arange(0, 290)

X_train = X[0:201, :]
X_test = X[200:, :]
y_train = y[0:201]
y_test = y[200:]
t_train = t[:201]
t_test = t[200:]


Parameters.CONSTANTS = [-1, 1]
Parameters.FEATURES = X_train.shape[1]
Parameters.CONSTANTS_TYPE = 'range'
# expressions = [Add, Sub, Mul, Div, Pow]
expressions = [Add, Sub, Mul, Div, Sin, Cos, Rlog, Exp, Pow]
terminals = [Variable, Constant]

max_time = '20'
method = 'FP'


model = DFP(pop_size=50,
        alpha=1e-4,
        beta=1e-2,
        gamma=1e-1,
        max_evaluations=100000,
        initial_min_depth=3,
        initial_max_depth=6,
        min_depth=3,
        max_depth=17,
        error_function=MSE,
        expressions=expressions,
        terminals=terminals,
        target_error=0,
        verbose=True
        )

model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

statistics = '{} {}\n'.format(
    mean_squared_error(y_train, y_fit),
    mean_squared_error(y_test, y_pred)
)

model.export_best()
print(statistics)


ax = plt.axes()
ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
plt.ylabel("BTC Price ($)")
line = [None, None, None, None]
line[0], = ax.plot(t_train, y_train, linestyle=':', color='black', linewidth=0.7, zorder=2, label='Targeted')    
line[1], = ax.plot(t_train, y_fit, linestyle='-', color='red', linewidth=0.7, zorder=3, label='Trained')
line[2], = ax.plot(t_test, y_test, linestyle=':', color='black', linewidth=0.7, zorder=2)
line[3], = ax.plot(t_test, y_pred, linestyle='-', color='blue', linewidth=0.7, zorder=3, label='Predicted')
plt.axvline(x=t_test[0], linestyle='-', color='black', linewidth='1')
plt.draw()
plt.legend()
plt.show()
# for i in range(30):

#     model = FP(pop_size=50,
#             alpha=0.01,
#             beta=0.05,
#             gamma=0.1,
#             max_evaluations=1000000,
#             max_time=20,
#             initial_min_depth=0,
#             initial_max_depth=6,
#             max_depth=15,
#             error_function=IR2,
#             expressions=expressions,
#             terminals=terminals,
#             target_error=0,
#             verbose=True
#             )

#     model.fit(X_train, y_train)
#     y_fit = model.predict(X_train)
#     y_pred = model.predict(X_test)

#     statistics = '{} {} {} {} {} {}\n'.format(
#         mean_squared_error(y_train, y_fit, squared=False),
#         mean_absolute_error(y_train, y_fit),
#         r2_score(y_train, y_fit),
#         mean_squared_error(y_test, y_pred, squared=False),
#         mean_absolute_error(y_test, y_pred),
#         r2_score(y_test, y_pred)
#     )

#     model.export_best(filename=str(i + 1))
#     print(statistics)

#     with open('tests/data/BTC/' + method + '/' + max_time + '/models.dat', 'a') as f1:
#         f1.write(model.best_individual.equation() + '\n')
#     with open('tests/data/BTC/' + method + '/' + max_time + '/records.dat', 'a') as f2:
#         f2.write(statistics)
    
#     ax = plt.axes()
#     ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
#     plt.ylabel("BTC Price ($)")
#     line = [None, None, None, None]
#     line[0], = ax.plot(date_train, y_train, linestyle=':', color='black', linewidth=0.7, zorder=2, label='Targeted')    
#     line[1], = ax.plot(date_train, y_fit, linestyle='-', color='red', linewidth=0.7, zorder=3, label='Trained')
#     line[2], = ax.plot(date_test, y_test, linestyle=':', color='black', linewidth=0.7, zorder=2)
#     line[3], = ax.plot(date_test, y_pred, linestyle='-', color='blue', linewidth=0.7, zorder=3, label='Predicted')
#     plt.axvline(x=date_test.iloc[0], linestyle='-', color='black', linewidth='1')
#     plt.draw()
#     plt.legend()
#     plt.show()