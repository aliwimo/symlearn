---
layout: default
title: API Reference
parent: Documentation
nav_order: 5
---

# [](#header-1)API Reference
{: .no_toc }

- TOC
{:toc}

## [](#header-2)Genetic Programming (GP) 
{: .d-inline-block }
New (V 0.1.1)
{: .label .label-purple }

Genetic programming (`GP`) is a type of evolutionary algorithm that uses a process similar to natural selection to evolve computer programs. In GP, a population of candidate solutions is evolved over generations using operations such as selection, mutation, and crossover, similar to how biological evolution works.


**Parameters:**

- `pop_size` (int): The size of trees population. Default is 100.
- `initial_min_depth` (int): The minimum depth of the trees during initializing them. Default is 0.
- `initial_max_depth` (int): The maximum depth of the trees during initializing them. Default is 6.
- `min_depth` (int): The minimum depth of the trees. Default is 1.
- `max_depth` (int): The maximum depth of the trees. Default is 15.
- `error_function` (function): The error function used to calculate the fitness of the trees.
- `expressions` (list): A list of functions that can be used as functional nodes in the trees. Default is [Add, Sub, Mul].
- `terminals` (list): A list of functions that can be used as leaf nodes in the trees. Default is [`Variable`, `Constant`].
- `target_error` (float): The target error for the model. Default is 0.0.
- `max_evaluations` (int): The maximum number of times the error function can be evaluated. Default is 10000.
- `max_generations` (int): The maximum number of generations that the model can run for. Default is -1, which means no maximum number of generations.
- `max_time` (int): The maximum amount of time the model can run for, in seconds. Default is None, which means no maximum time.
- `elite_number` (int): The number of elite individuals Default is 2.
- `crossover_probability` (float): The crossover probability. Default is 0.9.
- `mutation_probability` (float): The mutation probability. Default is 0.1.
- `tournament_size` (int): The tournament size. Default is 4.
- `selection_method` (str): The selection method (`wheel` or `tournament`). Default is `tournament`.
- `verbose` (bool): A flag indicating whether to print progress messages during model fitting. Default is True.

**Example:**
```python
from symlearn.models import GP
model = GP(
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
# fit data into model
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)
```

---

## [](#header-2)Firefly Programming (FFP)
Firefly programming (`FFP`) is an automatic programming method that uses the principles of symbolic regression to solve problems [^1].


**Parameters:**

- `pop_size` (int): The size of trees population. Default is 100.
- `initial_min_depth` (int): The minimum depth of the trees during initializing them. Default is 0.
- `initial_max_depth` (int): The maximum depth of the trees during initializing them. Default is 6.
- `min_depth` (int): The minimum depth of the trees. Default is 1.
- `max_depth` (int): The maximum depth of the trees. Default is 15.
- `error_function` (function): The error function used to calculate the fitness of the trees.
- `expressions` (list): A list of functions that can be used as functional nodes in the trees. Default is [Add, Sub, Mul].
- `terminals` (list): A list of functions that can be used as leaf nodes in the trees. Default is [`Variable`, `Constant`].
- `target_error` (float): The target error for the model. Default is 0.0.
- `max_evaluations` (int): The maximum number of times the error function can be evaluated. Default is 10000.
- `max_generations` (int): The maximum number of generations that the model can run for. Default is -1, which means no maximum number of generations.
- `max_time` (int): The maximum amount of time the model can run for, in seconds. Default is None, which means no maximum time.
- `verbose` (bool): A flag indicating whether to print progress messages during model fitting. Default is True.

**Example:**
```python
from symlearn.models import FFP
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
```

---

## [](#header-2)Difference-based Firefly Programming (DFFP)
Difference-based Firefly programming (`DFFP`) is an extended version of the standard firefly programming (`FFP`) [^2].

**Parameters:**

- `pop_size` (int): The size of trees population. Default is 100.
- `alpha` (float): Alpha controlling argument value. Default is 0.1.
- `beta` (float): Beta controlling argument value. Default is 0.5.
- `gamma` (float): Gamma controlling argument value. Default is 1.5.
- `initial_min_depth` (int): The minimum depth of the trees during initializing them. Default is 0.
- `initial_max_depth` (int): The maximum depth of the trees during initializing them. Default is 6.
- `min_depth` (int): The minimum depth of the trees. Default is 1.
- `max_depth` (int): The maximum depth of the trees. Default is 15.
- `error_function` (function): The error function used to calculate the fitness of the trees.
- `expressions` (list): A list of functions that can be used as functional nodes in the trees. Default is [Add, Sub, Mul].
- `terminals` (list): A list of functions that can be used as leaf nodes in the trees. Default is [`Variable`, `Constant`].
- `target_error` (float): The target error for the model. Default is 0.0.
- `max_evaluations` (int): The maximum number of times the error function can be evaluated. Default is 10000.
- `max_generations` (int): The maximum number of generations that the model can run for. Default is -1, which means no maximum number of generations.
- `max_time` (int): The maximum amount of time the model can run for, in seconds. Default is None, which means no maximum time.
- `verbose` (bool): A flag indicating whether to print progress messages during model fitting. Default is True.

**Example:**
```python
from symlearn.models import DFFP
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
```

---

## [](#header-2)Immune Plasma Programming (IPP)
Immune Plasma Programming (`IPP`) is an automatic programming method that uses the principles of symbolic regression to solve regression problems.

**Parameters:**

- `pop_size` (int): The size of trees population. Default is 100.
- `donors_number` (int): The number of donors. Default is 1.
- `receivers_number` (int): The number of receivers. Default is 1.
- `initial_min_depth` (int): The minimum depth of the trees during initializing them. Default is 0.
- `initial_max_depth` (int): The maximum depth of the trees during initializing them. Default is 6.
- `min_depth` (int): The minimum depth of the trees. Default is 1.
- `max_depth` (int): The maximum depth of the trees. Default is 15.
- `error_function` (function): The error function used to calculate the fitness of the trees.
- `expressions` (list): A list of functions that can be used as functional nodes in the trees. Default is [Add, Sub, Mul].
- `terminals` (list): A list of functions that can be used as leaf nodes in the trees. Default is [`Variable`, `Constant`].
- `target_error` (float): The target error for the model. Default is 0.0.
- `max_evaluations` (int): The maximum number of times the error function can be evaluated. Default is 10000.
- `max_generations` (int): The maximum number of generations that the model can run for. Default is -1, which means no maximum number of generations.
- `max_time` (int): The maximum amount of time the model can run for, in seconds. Default is None, which means no maximum time.
- `verbose` (bool): A flag indicating whether to print progress messages during model fitting. Default is True.

**Example:**
```python
from symlearn.models import IPP
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
```

---
## [](#header-3)References
{: .no_toc }

[^1]: https://ieeexplore.ieee.org/abstract/document/9302201/
[^2]: https://www.sciencedirect.com/science/article/abs/pii/S092054892300003X