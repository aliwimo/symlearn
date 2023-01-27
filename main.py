"""Main module that is used to test package"""

# import dependencies
import numpy as np

# import core dependencies
from symlearn.core.parameters import Parameters
from symlearn.core.functions import *
from symlearn.core.errors import *

# import models
from symlearn.models import FFP, DFFP, IPP, GP

# suppress numpy warnings
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

# print("FFP")
# model = FFP(pop_size=50,
#         max_evaluations=2500,
#         initial_min_depth=0,
#         initial_max_depth=6,
#         min_depth=1,
#         max_depth=15,
#         error_function=sum_of_difference,
#         expressions=expressions,
#         terminals=terminals,
#         target_error=0,
#         verbose=False
#         )

# # fit data into model
# model.fit(X_train, y_train)
# y_fit = model.predict(X_train)
# y_pred = model.predict(X_test)

# # print results of the model
# train_score = sum_of_difference(y_train, y_fit)
# test_score = sum_of_difference(y_test, y_pred)
# print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

# print("\nDFFP")
# model = DFFP(pop_size=50,
#         alpha=0.01,
#         beta=0.05,
#         gamma=0.1,
#         max_evaluations=5000,
#         initial_min_depth=0,
#         initial_max_depth=6,
#         min_depth=1,
#         max_depth=15,
#         error_function=sum_of_difference,
#         expressions=expressions,
#         terminals=terminals,
#         target_error=0,
#         verbose=False
#         )

# # fit data into model
# model.fit(X_train, y_train)
# y_fit = model.predict(X_train)
# y_pred = model.predict(X_test)

# # print results of the model
# train_score = sum_of_difference(y_train, y_fit)
# test_score = sum_of_difference(y_test, y_pred)
# print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

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


# print("\nGP")
# model = GP(pop_size=50,
#         max_evaluations=2500,
#         initial_min_depth=0,
#         initial_max_depth=6,
#         min_depth=1,
#         max_depth=15,
#         error_function=sum_of_difference,
#         expressions=expressions,
#         terminals=terminals,
#         target_error=0,
#         verbose=True
#         )

# # fit data into model
# model.fit(X_train, y_train)
# y_fit = model.predict(X_train)
# y_pred = model.predict(X_test)

# # print results of the model
# train_score = sum_of_difference(y_train, y_fit)
# test_score = sum_of_difference(y_test, y_pred)
# print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')
