import numpy as np
from copy import deepcopy
from random import random
from datetime import datetime, timedelta
from symlearn.core.methods import Methods
from symlearn.core.operators import share
from symlearn.core.functions import *
from symlearn.models.model import Model


class FFP(Model):
    """
    A class for that represents Firefly Programming (FFP) algorithm. This algorithm works by selecting the best
    tree at each generation and applying different operators to it to create new trees.
    """

    def __init__(self,
                 pop_size=100,
                 max_evaluations=10000,
                 max_generations=-1,
                 max_time=None,
                 initial_min_depth=0,
                 initial_max_depth=6,
                 min_depth=1,
                 max_depth=15,
                 error_function=None,
                 expressions=[Add, Sub, Mul],
                 terminals=[Variable, Constant],
                 target_error=0.0,
                 verbose=True
                 ):
        """Initializes the FFP algorithm."""
        super(FFP, self).__init__(
            pop_size=pop_size,
            max_evaluations=max_evaluations,
            max_generations=max_generations,
            max_time=max_time,
            initial_min_depth=initial_min_depth,
            initial_max_depth=initial_max_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            error_function=error_function,
            expressions=expressions,
            terminals=terminals,
            target_error=target_error,
            verbose=verbose)

    def fit(self, X, y):
        """
        Trains the model on the given data.

        Args:
            X (numpy array): The input data.
            y (numpy array): The target values.

        Returns:
            None
        """
        self.X = X
        self.y = y
        if self.max_time:
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(seconds=self.max_time)
        self._generate_population()
        self._get_initial_statistics()
        self._run()
        if self.verbose:
            if self.max_time:
                print(f'Total time: {datetime.now() - self.start_time}')
            print(f'Evaluations: {self.current_evaluation}')

    def _attract(self, i, j):
        """
        Applies the "share" operator to the trees at indices `i` and `j` in the population.

        Args:
            i (int): The index of the first tree.
            j (int): The index of the second tree.

        Returns:
            Tree: The resulting tree after the "share" operator has been applied.
        """
        return share(self.population[j], deepcopy(self.population[i]))

    def _run(self):
        """
        Runs the Firefly Programming (FFP) algorithm until termination conditions are met.

        Returns:
            None
        """
        while not self._should_terminate():
            self._rank(is_reversed=False)
            for i in range(self.pop_size):
                if self._should_terminate():
                    break
                for j in range(self.pop_size):
                    if self._should_terminate():
                        break
                    if self.population[i].fitness >= self.population[j].fitness:
                        temp = self._attract(i, j)
                        if temp.depth() > self.max_depth or temp.depth() < self.min_depth:
                            if random() > 0.5:
                                temp = Methods.generate_individual(
                                    'full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                            else:
                                temp = Methods.generate_individual(
                                    'grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                        self._evaluate(i, temp)
                        self.current_evaluation += 1

                    if self._should_terminate():
                        break
                if self._should_terminate():
                    break
            if self._should_terminate():
                break
            # increase generation counter
            if not self.max_generations == -1:
                self.current_generation += 1
