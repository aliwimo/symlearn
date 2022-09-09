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
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import os
from pathlib import Path

np.seterr(all='ignore')

# df = pd.read_csv('data/btc_usd.csv')
df = pd.read_csv('data/BTC-USD2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['PastDay'] = df['Close'].shift(1)
# df['PastDayHigh'] = df['High'].shift(1)
# df['PastDayLow'] = df['Low'].shift(1)

train_mask = (df['Date'] >= '2021-9-1') & (df['Date'] <= '2022-5-27')
train_df = df.loc[train_mask]

test_mask = (df['Date'] >= '2022-5-28') & (df['Date'] <= '2022-8-28')
test_df = df.loc[test_mask]

X_train = train_df[['PastDay', 'High', 'Low']]
# X_train = train_df[['PastDay', 'PastDayHigh', 'PastDayLow']]
y_train = train_df['Close']
date_train = train_df['Date']
X_test = test_df[['PastDay', 'High', 'Low']]
# X_test = test_df[['PastDay', 'PastDayHigh', 'PastDayLow']]
y_test = test_df['Close']
date_test = test_df['Date']

# convert to numpy objects
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

Parameters.CONSTANTS = [-5, 5]
Parameters.FEATURES = X_train.shape[1]
Parameters.CONSTANTS_TYPE = 'range'
expressions = [Add, Sub, Mul, Div]
terminals = [Variable, Constant]

evals = '07'
method = 'SKL'
export_data = False

if export_data:
    working_dir = Path(__file__).parent.absolute()
    export_path = 'temp/BTC/' + method + '/' + evals + '/'
    dirpath = str(working_dir) + '/' + export_path
    if not os.path.exists(dirpath):
        os.makedirs(dirpath + 'graphs/')  


for i in range(0, 1):

    model = FP(pop_size=50,
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
            verbose=True
            )

    
    # model = MLPRegressor(max_iter=5000, hidden_layer_sizes=(4, 4, 4, ))
    # model = tree.DecisionTreeRegressor()
    # model = LinearRegression()
    # model = KNeighborsRegressor(n_neighbors=2)
    # model = make_pipeline(StandardScaler(), SVR(C=100.0, coef0=1.0, kernel='poly', max_iter=5000))

    

    model.fit(X_train, y_train)
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)

    # statistics = '{} {} {} {} {} {}'.format(
    statistics = '{} {}'.format(
        # mean_squared_error(y_train, y_fit, squared=False),
        # mean_absolute_error(y_train, y_fit),
        r2_score(y_train, y_fit),
        # mean_squared_error(y_test, y_pred, squared=False),
        # mean_absolute_error(y_test, y_pred),
        r2_score(y_test, y_pred)
    )
    

    iteration = str("{0:0=2d}".format(i + 1))
    print(f'Iteration: {iteration}')
    print(statistics)

    if export_data:
        # model.export_best(export_path=export_path + '/trees/', filename=iteration)
        # with open(dirpath + '/models.dat', 'a') as f1:
        #     f1.write(model.best_individual.equation() + '\n')
        with open(dirpath + '/records.dat', 'a') as f2:
            f2.write(statistics + '\n')

    # graph_path = dirpath + 'graphs/' + iteration + '.' + Parameters.EXPORT_EXT
    # print(graph_path)
    # print('-' * 100)

    plt.clf()
    ax = plt.axes()
    ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
    plt.ylabel("BTC Price ($)")
    line = [None, None, None, None]
    line[0], = ax.plot(date_train, y_train, linestyle=':', color='black', linewidth=0.7, zorder=2, label='Targeted')    
    line[1], = ax.plot(date_train, y_fit, linestyle='-', color='red', linewidth=0.7, zorder=3, label='Trained')
    line[2], = ax.plot(date_test, y_test, linestyle=':', color='black', linewidth=0.7, zorder=2)
    line[3], = ax.plot(date_test, y_pred, linestyle='-', color='blue', linewidth=0.7, zorder=3, label='Predicted')
    plt.axvline(x=date_test.iloc[0], linestyle='-', color='black', linewidth='1')
    plt.draw()
    plt.legend()
    if export_data:
        fig = plt.gcf()
        fig.set_size_inches(13.66, 6.66)
        fig.savefig(graph_path, dpi=100)
    else:
        plt.show()