# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import core dependicies 
from core.parameters import Parameters
from core.functions import *
from core.errors import *

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

# import models
from models.ffp import FFP
from models.dffp import DFFP
from models.ipp import IPP

# supress numpy warnings
np.seterr(all='ignore')

# import dataset
df = pd.read_csv('datasets/btc-usd.csv')

# convert date column to date format
df['Date'] = pd.to_datetime(df['Date'])

# split dataset into training and test subsets
train_mask = (df['Date'] >= '2022-1-2') & (df['Date'] <= '2022-8-31')
train_df = df.loc[train_mask]
test_mask = (df['Date'] >= '2022-9-1') & (df['Date'] <= '2022-10-31')
test_df = df.loc[test_mask]

# split each of training and test subsets into inputs (X) and outputs (Y)
X_train = train_df[['Open', 'High', 'Low']]
y_train = train_df['Close']
date_train = train_df['Date']
X_test = test_df[['Open', 'High', 'Low']]
y_test = test_df['Close']
date_test = test_df['Date']

# convert dataframes to numpy objects
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

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
        error_function=IR2,
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
print('Training set r2 score: {}\nTest set r2 score: {}'.format(
    r2_score(y_train, y_fit),
    r2_score(y_test, y_pred)
))

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
        error_function=IR2,
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
print('Training set r2 score: {}\nTest set r2 score: {}'.format(
    r2_score(y_train, y_fit),
    r2_score(y_test, y_pred)
))

print("\nIPP")
model = IPP(pop_size=100,
            donors_number=3,
            receivers_number=3,
            max_evaluations=2500,
            initial_min_depth=0,
            initial_max_depth=6,
            min_depth=1,
            max_depth=15,
            error_function=IR2,
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
print('Training set r2 score: {}\nTest set r2 score: {}'.format(
    r2_score(y_train, y_fit),
    r2_score(y_test, y_pred)
))

# # plot model graph
# plt.clf()
# ax = plt.axes()
# ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
# plt.ylabel("BTC Price ($)")
# line = [None, None, None, None]
# line[0], = ax.plot(date_train, y_train, linestyle=':',
#                    color='black', linewidth=0.7, zorder=2, label='Targeted')
# line[1], = ax.plot(date_train, y_fit, linestyle='-',
#                    color='red', linewidth=0.7, zorder=3, label='Trained')
# line[2], = ax.plot(date_test, y_test, linestyle=':',
#                    color='black', linewidth=0.7, zorder=2)
# line[3], = ax.plot(date_test, y_pred, linestyle='-',
#                    color='blue', linewidth=0.7, zorder=3, label='Predicted')
# plt.axvline(x=date_test.iloc[0], linestyle='-', color='black', linewidth='1')
# plt.draw()
# plt.legend()
# plt.show()
