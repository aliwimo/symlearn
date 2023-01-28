import numpy as np
from copy import deepcopy
from random import random, randint
from datetime import datetime, timedelta
from symlearn.core.methods import Methods
from symlearn.core.operators import share
from symlearn.core.functions import *
from symlearn.models.model import Model


class GP(Model):
    """
    A class for that represents Genetic Programming (GP) algorithm. This algorithm works by selecting the best
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
                 n_elite=2,
                 p_crossover=0.9,
                 verbose=True
                 ):
        """Initializes the GP algorithm."""
        super(GP, self).__init__(
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
        self.n_elite = n_elite
        self.p_crossover = p_crossover

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

    
    # roulette wheel selection
    def perform_selection(self):
        return self.rand_selection()

    # def wheel_selection(self):
    #     total_fit = sum(self.fitnesses)
    #     prop = [(i / total_fit) for i in fitnesses]
    #     chosen = np.random.choice(chromosome_number, p=prop, size=2) # 
    #     return population[chosen[0]], population[chosen[1]]
    
    def rand_selection(self):
        index1 = randint(0, self.pop_size - 1)
        index2 = randint(0, self.pop_size - 1)
        return self.population[index1], self.population[index2]

    def perform_crossover(self, parent1, parent2):
        child1 = share(deepcopy(parent2), deepcopy(parent1))
        child2 = share(deepcopy(parent1), deepcopy(parent2))
        if child1.depth() > self.max_depth or child1.depth() < self.min_depth:
            child1 = self._generate_random_tree()
        if child2.depth() > self.max_depth or child2.depth() < self.min_depth:
            child2 = self._generate_random_tree()
        return child1, child2

    def _run(self):
        """
        Runs the Genetic Programming (GP) algorithm until termination conditions are met.

        Returns:
            None
        """
        while not self._should_terminate():
            next_generation = [0] * self.pop_size
    
            # we have to resort population according to their fitnesses
            self._rank(is_reversed=False)

            # copy elites to the new generation
            for i in range(self.n_elite): 
                next_generation[i] = deepcopy(self.population[i])

            # perform crossover
            for i in range(self.n_elite, self.pop_size, 2):
                parent1, parent2 = self.perform_selection()
                if random() < self.p_crossover:
                    child1, child2 = self.perform_crossover(parent1, parent2)
                    next_generation[i] = deepcopy(child1)
                    next_generation[i + 1] = deepcopy(child2)
                else:
                    next_generation[i] = deepcopy(parent1)
                    next_generation[i + 1] = deepcopy(parent2)
            
            # update and calculate the fitnesses of the next generation and find the best
            self._update_errors(next_generation)
            next_fitnesses = self._calculate_errors(next_generation)
            next_generation_best = self._find_best_model(next_generation, next_fitnesses)


            for i in range(self.pop_size):
                if next_fitnesses[i] > self.fitnesses[i]:
                    self.population[i] = deepcopy(next_generation[i])
                    self.fitnesses[i] = next_fitnesses[i]

            # increase evaluation counter
            self.current_evaluation += self.pop_size

            self._compare_model(next_generation_best)
            
            # increase generation counter
            if not self.max_generations == -1:
                self.current_generation += 1
