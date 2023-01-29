---
layout: default
title: Metrics
parent: Documentation
nav_order: 4
mathjax: true
tags: 
  - latex
  - math
---

# [](#header-1)Metrics
{: .no_toc }

Since `symlearn` uses optimization algorithms, it needs metrics to act as an objective (error) function to optimize models. Therefore, it is necessary to define metric functions. `symlearn` package provides most popular regression metrics listed below:

- TOC
{:toc}

## [](#header-2) Sum of Absolute Error (SAE)
The Sum of Absolute Error (SAE) is a commonly used regression metric that measures the difference between the predicted values and the actual values. It is defined as the sum of the absolute differences between the predicted and actual values for each data point. Mathematically, it can be represented as:

$$ SAE = \sum_{i=1}^{n} (|y_{i} - {\hat{y}}_{i}|) $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, and $$n$$ is the number of data points.

It is worth noting that SAE is sensitive to outliers and can be affected by large errors.

**Usage:** `sum_absolute_difference`

```python
# import metric
from symlearn.core.metrics import sum_absolute_difference
```
```python
# initialize models
ffp_model = FFP(
            error_function=sum_absolute_difference,
            # arguments
            )
```
```python
# calculate scores
train_score = sum_absolute_difference(y_train, y_fit)
test_score = sum_absolute_difference(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

## [](#header-2) Mean Squared Error (MSE)
The Mean Squared Error (MSE) is a commonly used regression metric that measures the average of the squared differences between the predicted values and the actual values. It is defined as the average of the squared differences between the predicted and actual values for each data point. Mathematically, it can be represented as:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{i} - {\hat{y}}_{i})^2 $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, and $$n$$ is the number of data points.

MSE is widely used because it penalizes large errors more than smaller errors and it is differentiable and can be used to optimize models using optimization algorithms. However, MSE is sensitive to outliers and the unit of measurement is squared of the target variable.

**Usage:** `mean_squared_error`

```python
# import metric
from symlearn.core.metrics import mean_squared_error
```
```python
# initialize models
ffp_model = FFP(
            error_function=mean_squared_error,
            # arguments
            )
```
```python
# calculate scores
train_score = mean_squared_error(y_train, y_fit)
test_score = mean_squared_error(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

## [](#header-2) Root Mean Squared Error (RMSE)
The Root Mean Squared Error (RMSE) is a commonly used regression metric that measures the average of the squared differences between the predicted values and the actual values and then taking the square root of the result. It is defined as the square root of the average of the squared differences between the predicted and actual values for each data point. Mathematically, it can be represented as:

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{i} - {\hat{y}}_{i})^2} $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, and $$n$$ is the number of data points.

RMSE is similar to MSE and it is differentiable and can be used to optimize models using optimization algorithms. However, the difference is that RMSE is in the same unit of measurement as the target variable. Because RMSE is in the same unit of measurement as the target variable, it is easier to interpret and less sensitive to outliers than MSE.

**Usage:** `root_mean_squared_error`

```python
# import metric
from symlearn.core.metrics import root_mean_squared_error
```
```python
# initialize models
ffp_model = FFP(
            error_function=root_mean_squared_error,
            # arguments
            )
```
```python
# calculate scores
train_score = root_mean_squared_error(y_train, y_fit)
test_score = root_mean_squared_error(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

## [](#header-2) Mean Absolute Error (MAE)
The Mean Absolute Error (MAE) is a commonly used regression metric that measures the average difference between the predicted values and the actual values. It is defined as the average of the absolute differences between the predicted and actual values for each data point. Mathematically, it can be represented as:

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} (|y_{i} - {\hat{y}}_{i}|) $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, and $$n$$ is the number of data points.

MAE is less sensitive to outliers than other metrics like Mean Squared Error (MSE) and it is easy to interpret because it has the same unit of measurement as the target variable.

**Usage:** `mean_absolute_error`

```python
# import metric
from symlearn.core.metrics import mean_absolute_error
```
```python
# initialize models
ffp_model = FFP(
            error_function=mean_absolute_error,
            # arguments
            )
```
```python
# calculate scores
train_score = mean_absolute_error(y_train, y_fit)
test_score = mean_absolute_error(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

## [](#header-2) R-squared Score (R<sup>2</sup>)
The R-squared ($$R^{2}$$) score, also known as the coefficient of determination, is a commonly used regression metric that measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from $$-\infty$$ to 1, where a higher value indicates a better fit of the model to the data. Mathematically, it can be represented as:

$$ R^{2} = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_{i} - {\hat{y}}_{i})^{2}}{\sum_{i=1}^{n} ((y_{i} - \overline{y})^{2})} $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, $$\overline{y}$$ is the mean of the actual values, and $$n$$ is the number of data points.

$$R^{2}$$ is a commonly used metric because it is easy to interpret and it tells how much of the variance in the dependent variable is explained by the independent variable(s) and it is independent of the scale of the dependent variable. But, it can be misleading if a model with a high $$R^{2}$$ score is not actually a good model because it may overfit the data.

**Usage:** `r2_score`

```python
# import metric
from symlearn.core.metrics import r2_score
```
```python
# calculate scores
train_score = r2_score(y_train, y_fit)
test_score = r2_score(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

{: .warning }
`r2_score` cannot be used as an `error_function` while initializing models becuase optimization process in `symlearn` is an minimization process. Therefore, if you want to use a metric that the higher values is better than the lower values, you have to inverse it. In case of `r2_score`, you can use `r2_score_inverse` metric instead as an `error_function`.

## [](#header-2) Inversed R-squared Score (IR<sup>2</sup>)
The inverse R-squared ($$IR^2$$) score is not a commonly used regression metric. It is the reciprocal of R-squared score, which is a commonly used regression metric that measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to $$\infty$$, where a lower value indicates a better fit of the model to the data. Mathematically, it can be represented as:

$$ IR^{2} = 1 - R^{2} = \frac{\sum_{i=1}^{n} (y_{i} - {\hat{y}}_{i})^{2}}{\sum_{i=1}^{n} ((y_{i} - \overline{y})^{2})} $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, $$\overline{y}$$ is the mean of the actual values, and $$n$$ is the number of data points.

It is important to note that the inverse R-squared score is not a commonly used metric, and it is not a commonly accepted way to evaluate the performance of a regression model. It may be useful in certain contexts, but it should be used with caution. It is not always meaningful to interpret it.

**Usage:** `r2_score_inverse`

```python
# import metric
from symlearn.core.metrics import r2_score_inverse
```
```python
# initialize models
ffp_model = FFP(
            error_function=r2_score_inverse,
            # arguments
            )
```
```python
# calculate scores
train_score = r2_score_inverse(y_train, y_fit)
test_score = r2_score_inverse(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```

## [](#header-2) Maximum Residual Error (MAX ERROR)
The maximum residual error metric in regression is a measure of the largest difference between the predicted value of the model and the actual value of the dependent variable. It is often used as a measure of the worst-case error of a model. Mathematically, it can be represented as:

$$ MAX\;ERROR = max(|y_{i} - {\hat{y}}_{i}|) \quad , \quad \forall i \in n $$

where $$y_{i}$$ is the actual value of the $$i^{th}$$ data point, $${\hat{y}}_{i}$$ is the predicted value of the $$i^{th}$$ data point, $$\overline{y}$$ is the mean of the actual values, and $$n$$ is the number of data points.

The maximum residual error metric is useful for identifying the worst-case error of a model, but it can be sensitive to outliers and may not always be representative of the overall performance of the model. It is best used in conjunction with other metrics like mean absolute error, mean squared error, and R-squared score to get a better understanding of the model's performance.

**Usage:** `max_residual_error`

```python
# import metric
from symlearn.core.metrics import max_residual_error
```
```python
# initialize models
ffp_model = FFP(
            error_function=max_residual_error,
            # arguments
            )
```
```python
# calculate scores
train_score = max_residual_error(y_train, y_fit)
test_score = max_residual_error(y_test, y_pred)
print(f'Training Set Score: {train_score}')
print(f'Testing Set Score: {test_score}')
```
