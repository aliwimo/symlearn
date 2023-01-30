import numpy as np
from copy import deepcopy, copy
from random import random, randint, choices
from datetime import datetime, timedelta
from symlearn.core.methods import Methods
from symlearn.core.operators import share, substitute
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
                 elite_number=2,
                 crossover_probability=0.9,
                 mutation_probability=0.1,
                 tournament_size=4,
                 selection_method='tournament', # possible choices: wheel, tournament
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

        self.elite_number = elite_number
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        if selection_method == 'tournament':
            self.selection_method = self.tournament_selection
        elif selection_method == 'wheel':
            self.selection_method = self.wheel_selection
        

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

    
    def perform_selection(self):
        return self.selection_method()

    def wheel_selection(self):
        # since its a minimization problem, 
        # its necessary to get the inverse values of fitnesses
        fitnesses_inverse = 1 / np.array(self.fitnesses)
        total_sum = sum(fitnesses_inverse)
        probabilities = [(i / total_sum) for i in fitnesses_inverse]
        chosen = np.random.choice(self.pop_size, p=probabilities, size=2)
        return self.population[chosen[0]], self.population[chosen[1]]

    def tournament_selection(self):
        parents = [0] * self.tournament_size
        parents_fitnesses = [0] * self.tournament_size
        indexes = choices(range(self.pop_size), k=self.tournament_size)
        for i in range(self.tournament_size):
            parents[i] = deepcopy(self.population[indexes[i]])
            parents_fitnesses[i] = self.fitnesses[indexes[i]].copy()
        parents, parents_fitnesses = Methods.rank_trees(self.population, self.fitnesses)
        return parents[0], parents[1]

    def perform_mutation(self, tree):
        tree = substitute(tree, self.expressions + self.terminals)
        return self._check_depth(tree)


    def rand_selection(self):
        index1 = randint(0, self.pop_size - 1)
        index2 = randint(0, self.pop_size - 1)
        return self.population[index1], self.population[index2]

    def perform_crossover(self, parent1, parent2):
        child1 = share(deepcopy(parent2), deepcopy(parent1))
        child2 = share(deepcopy(parent1), deepcopy(parent2))
        # check depth after sharing
        child1 = self._check_depth(child1)
        child2 = self._check_depth(child2)
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
            for i in range(self.elite_number): 
                next_generation[i] = deepcopy(self.population[i])

            # perform crossover
            for i in range(self.elite_number, self.pop_size, 2):
                parent1, parent2 = self.perform_selection()
                if random() < self.crossover_probability:
                    child1, child2 = self.perform_crossover(parent1, parent2)
                    next_generation[i] = deepcopy(child1)
                    next_generation[i + 1] = deepcopy(child2)
                else:
                    next_generation[i] = deepcopy(parent1)
                    next_generation[i + 1] = deepcopy(parent2)

            # perform mutation
            for i in range(self.pop_size):
                if random() < self.mutation_probability:
                    next_generation[i] = self.perform_mutation(deepcopy(self.population[i]))

            # update and calculate the fitnesses of the next generation and find the best
            self._update_errors(next_generation)
            next_fitnesses = self._calculate_errors(next_generation)
            next_generation_best = self._find_best_model(next_generation, next_fitnesses)

            # comparing current and next generations
            for i in range(self.pop_size):
                if next_fitnesses[i] > self.fitnesses[i]:
                    self.population[i] = deepcopy(next_generation[i])
                    self.fitnesses[i] = next_fitnesses[i]

            # increase evaluation counter
            self.current_evaluation += self.pop_size
            
            # compare with best model
            self._compare_model(next_generation_best)
            
            # increase generation counter
            if not self.max_generations == -1:
                self.current_generation += 1
