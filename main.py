"""Main module that is used to test package"""

# import dependencies
import numpy as np
import pandas as pd

# import core dependencies
from symlearn.core.parameters import Parameters
from symlearn.core.functions import *
from symlearn.core.metrics import *

# import models
from symlearn.models import FFP, DFFP, IPP, GP

# suppress numpy warnings
np.seterr(all='ignore')

# import dataset
data = pd.read_csv('example/data.csv').to_numpy()
X = data[:, [0, 1]]
y = data[:, 2]
t = np.arange(0, 290)
# split each of training and test subsets into inputs (X) and outputs (Y)
X_train = X[0:201, :]
X_test = X[200:, :]
y_train = y[0:201]
y_test = y[200:]
t_train = t[:201]
t_test = t[200:]

# set global parameters
Parameters.CONSTANTS = [-5, 5]
Parameters.FEATURES = X_train.shape[1]
Parameters.CONSTANTS_TYPE = 'range'
expressions = [Add, Sub, Mul, Div, Sin, Cos]
terminals = [Variable, Constant]

population_size = 10
max_evaluations = 1000
initial_min_depth = 0
initial_max_depth = 6
min_depth = 1
max_depth = 15
error_function = root_mean_squared_error
target_error = 0
verbose = False

# initialize models
ffp_model = FFP(
            pop_size=population_size,
            max_evaluations=max_evaluations,
            initial_min_depth=initial_min_depth,
            initial_max_depth=initial_max_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            error_function=error_function,
            expressions=expressions,
            terminals=terminals,
            target_error=target_error,
            verbose=verbose
            )

dffp_model = DFFP(
            pop_size=population_size,
            alpha=0.01,
            beta=0.05,
            gamma=0.1,
            max_evaluations=max_evaluations,
            initial_min_depth=initial_min_depth,
            initial_max_depth=initial_max_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            error_function=error_function,
            expressions=expressions,
            terminals=terminals,
            target_error=target_error,
            verbose=verbose
            )

ipp_model = IPP(
            pop_size=population_size,
            donors_number=3,
            receivers_number=3,
            max_evaluations=max_evaluations,
            initial_min_depth=initial_min_depth,
            initial_max_depth=initial_max_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            error_function=error_function,
            expressions=expressions,
            terminals=terminals,
            target_error=target_error,
            verbose=verbose
            )

gp_model = GP(
            pop_size=population_size,
            max_evaluations=max_evaluations,
            initial_min_depth=initial_min_depth,
            initial_max_depth=initial_max_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            error_function=error_function,
            expressions=expressions,
            terminals=terminals,
            target_error=target_error,
            verbose=verbose
            )


models = [ffp_model, dffp_model, ipp_model, gp_model]
# models = [ffp_model]
# models = [gp_model]

for model in models:
    print('-' * 50)
    # print name of the model
    print(model.__class__.__name__)

    # fit data into model
    model.fit(X_train, y_train)
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)

    # print results of the model
    train_score = error_function(y_train, y_fit)
    test_score = error_function(y_test, y_pred)
    print(f'Training set `{error_function.__name__}` score: {train_score}\nTest set `{error_function.__name__}` score: {test_score}')
