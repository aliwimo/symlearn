---
layout: default
title: How To Use
parent: Documentation
nav_order: 3
---

# [](#header-1)How to use?

After installing ``symlearn`` package, you can use it by importing the essential modules and classes. The following example shows how to use ``symlearn`` package in your projects. Example codes and data files can be found in [Package Repository](https://github.com/aliwimo/symlearn).

In this example, we used `FFP` is used in modeling and forecasting the system of Box-Jenkins gas furnace time series. It starts with importing dataset and splitting it into training and test subsets:

```python
# import dataset and split it into training and test subsets
box_dataset = pd.read_csv('datasets/box-jenkins.csv').to_numpy()
X = box_dataset[:, [0, 1]]
y = box_dataset[:, 2]
t = np.arange(0, 290)
X_train = X[0:201, :]
X_test = X[200:, :]
y_train = y[0:201]
y_test = y[200:]
t_train = t[:201]
t_test = t[200:]
```

After that symlearn's core functions and modules must be imported.

```python
# import core dependicies
from symlearn.core.parameters import Parameters
from symlearn.core.errors import sum_of_difference
from symlearn.core.functions import *
```

After Importing the essential core modules, ``Parameters`` must be declared before start of optimizing process.

```python
# set parameters, expressions, and terminals
Parameters.CONSTANTS = [-1, 1]
Parameters.FEATURES = X_train.shape[1]
Parameters.CONSTANTS_TYPE = 'range'
expressions = [Add, Sub, Mul, Div, Pow]
terminals = [Variable, Constant]
```

A suitable model must be chosen, and its arguments must be satisfied. Here in this example, we used `FFP` model with the arguments shown in the following code block:

```python
# import error metric, FFP model and initialize it     
from symlearn.models import FFP

model = FFP(pop_size=50,
        max_evaluations=5000,
        initial_min_depth=0,
        initial_max_depth=6,
        min_depth=1,
        max_depth=15,
        error_function=sum_of_difference,
        expressions=expressions,
        terminals=terminals,
        target_error=0,
        verbose=True
        )

```

Fitting model using `fit()` method.

```python
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)
```
<div class="code-example" markdown="1">
```cmd
Evaluations: 5 | Fitness: 14.163321775790926
Evaluations: 68 | Fitness: 8.194608
Evaluations: 823 | Fitness: 5.095621733489572
Evaluations: 1917 | Fitness: 4.3966582987110225
Evaluations: 2850 | Fitness: 3.7636233912595776
Evaluations: 3433 | Fitness: 3.561661510016762
Evaluations: 4148 | Fitness: 3.540966725348731
Evaluations: 5001
```
</div>

Plotting the optimized model

```python
# plotting the optimized model    
import matplotlib.pyplot as plt
ax = plt.axes()
ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
line = [None, None, None, None]
line[0], = ax.plot(t_train, y_train, linestyle=':', color='black', linewidth=0.7, zorder=2, label='Targeted')    
line[1], = ax.plot(t_train, y_fit, linestyle='-', color='red', linewidth=0.7, zorder=3, label='Trained')
line[2], = ax.plot(t_test, y_test, linestyle=':', color='black', linewidth=0.7, zorder=2)
line[3], = ax.plot(t_test, y_pred, linestyle='-', color='blue', linewidth=0.7, zorder=3, label='Predicted')
plt.axvline(x=t_test[0], linestyle='-', color='black', linewidth='1')
plt.draw()
plt.legend()
plt.show()
```

![](../../assets/images/graph.jpg)
*Targeted and Predicted models' graphs*

Exporting the graph of the best model:

```python
model.export_best()
```
<div class="code-example" markdown="1">
```cmd
((x2)-(((((x0)+(x1))-(-1.74))-(x1))+((((-1.12)*(1.06))-((x1)+(-3.78)))-(1.41))))
```
</div>

<img src="../../assets/images/exported_graph.jpg" width="400px"/>

*Best models' tree representation*
