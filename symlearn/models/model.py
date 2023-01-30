import numpy as np
from datetime import datetime
from random import random
from symlearn.core.methods import Methods
from symlearn.core.functions import *
from symlearn.core.metrics import r2_score
from copy import deepcopy


class Model:
    """
    This is the main class for the symbolic regression model. It contains all the basic and shared methods and attributes
    for fitting, predicting and evaluating the model.

    Args:
        pop_size (int): The size of trees population. Default is 100.
        initial_min_depth (int): The minimum depth of the trees during initializing them. Default is 0.
        initial_max_depth (int): The maximum depth of the trees during initializing them. Default is 6.
        min_depth (int): The minimum depth of the trees. Default is 1.
        max_depth (int): The maximum depth of the trees. Default is 15.
        error_function (function): The error function used to calculate the fitness of the trees.
        expressions (list): A list of functions that can be used as functional nodes in the trees. Default is [Add, Sub, Mul].
        terminals (list): A list of functions that can be used as leaf nodes in the trees. Default is [Variable, Constant].
        target_error (float): The target error for the model. Default is 0.0.
        max_evaluations (int): The maximum number of times the error function can be evaluated. Default is 10000.
        max_generations (int): The maximum number of generations that the model can run for. Default is -1, which means no maximum number of generations.
        max_time (int): The maximum amount of time the model can run for, in seconds. Default is None, which means no maximum time.
        verbose (bool): A flag indicating whether to print progress messages during model fitting. Default is True.
    """

    def __init__(self,
                 pop_size=100,
                 initial_min_depth=0,
                 initial_max_depth=6,
                 min_depth=1,
                 max_depth=15,
                 error_function=None,
                 expressions=[Add, Sub, Mul],
                 terminals=[Variable, Constant],
                 target_error=0.0,
                 max_evaluations=10000,
                 max_generations=-1,
                 max_time=None,
                 verbose=True
                 ) -> None:
        """Initializing method."""
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.current_evaluation = 0
        self.max_generations = max_generations
        self.current_generation = 0
        self.max_time = max_time
        self.start_time = None
        self.end_time = None
        self.initial_min_depth = initial_min_depth
        self.initial_max_depth = initial_max_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.error_function = error_function
        self.expressions = expressions
        self.terminals = terminals
        self.target_error = target_error
        self.verbose = verbose
        self.model = None
        self.population = None
        self.fitnesses = None

    def _generate_population(self):
        """
        Generates the initial population of trees.

        Returns:
            None
        """
        self.population = Methods.generate_population(
            pop_size=self.pop_size,
            initial_min_depth=self.initial_min_depth,
            initial_max_depth=self.initial_max_depth,
            expressions=self.expressions,
            terminals=self.terminals)

    def _get_initial_statistics(self):
        """
        Calculates the fitness of the trees in the initial population, 
        and sets the model to the tree with the lowest fitness.

        Returns:
            None
        """
        self._update_errors()
        self.fitnesses = self._calculate_errors()
        self.model = self._find_best_model()

    def _update_errors(self, population=None):
        if not population: population = self.population
        for tree in population: tree.update_fitness(self.error_function, self.X, self.y)

    def _calculate_errors(self, population=None):
        if not population: population = self.population
        return [tree.fitness for tree in population]

    def _find_best_model(self, population=None, errors=None):
        if not population: population = self.population
        if not errors: errors = self.fitnesses
        min_error_index = errors.index(min(errors))
        return deepcopy(population[min_error_index])
            
    def _rank(self, is_reversed=False):
        """
        Ranks the trees in the population according to their fitness.

        Args:
            is_reversed (bool): A flag indicating whether to sort in reverse order. Default is False.

        Returns:
            None
        """
        self.population, self.fitnesses = Methods.rank_trees(
            self.population, self.fitnesses, is_reversed)

    def _evaluate(self, current, temp):
        """
        Evaluates the fitness of the given tree, and updates the model if the tree has a lower fitness than the current model.

        Args:
            current (int): The index of the tree in the population.
            temp (Tree): The tree to be evaluated.

        Returns:
            None
        """
        temp.update_fitness(self.error_function, self.X, self.y)
        if temp.fitness < self.population[current].fitness:
            self.population[current] = deepcopy(temp)
            self.fitnesses[current] = self.population[current].fitness
            self._compare_model(temp)
    
    def _compare_model(self, tree):
        if tree.fitness < self.model.fitness:
            self.model = deepcopy(tree)
            self.print_info()

    def print_info(self):
        if self.verbose:
            print(f'Evaluations: {self.current_evaluation} | Fitness: {self.model.fitness}')

    def score(self, y_test, y_pred):
        """
        Calculates the R^2 score of the model.

        Args:
            y_test (numpy array): The true values of the target variable.
            y_pred (numpy array): The predicted values of the target variable.

        Returns:
            float: The R^2 score of the model.
        """
        return r2_score(y_test, y_pred)

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (numpy array): The input data.

        Returns:
            numpy array: The predictions made by the model.
        """
        return self.model.output(X)

    def _should_terminate(self):
        """
        Determines whether the model fitting process should terminate based on the specified termination criteria.

        Returns:
            bool: A flag indicating whether the model fitting process should terminate.
        """
        if self.max_time and datetime.now() > self.end_time:
            return True
        if self.max_evaluations > -1 and self.current_evaluation >= self.max_evaluations:
            return True
        if self.max_generations > -1 and self.current_generation >= self.max_generations:
            return True
        if self.model.fitness < self.target_error:
            return True
        return False

    def export_best(self, export_path='images/', filename='Best'):
        """
        Exports a graphical representation of the best tree in the population to a file.

        Args:
            export_path (str): The path to the directory where the file should be saved. Default is 'images/'.
            filename (str): The name of the file. Default is 'Best'.

        Returns:
            None
        """
        if self.model:
            label = "Best error: "
            label += str(round(self.model.fitness, 3))
            Methods.export_graph(self.model, export_path + filename, label)
            print(self.model.equation())

    def test_model(self, X):
        """
        Tests the model using the given data.

        Args:
            X (numpy array): The input data to be used for testing.

        Returns:
            numpy array: The predictions made by the model on the given data.
        """
        zero_array = np.zeros(X.shape)
        x = np.vstack([zero_array, X])
        return self.model.output(x)[-1]

    def _generate_random_tree(self):
        method = 'full' if random() > 0.5 else 'grow'
        return Methods.generate_individual(method, self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
        
    def _check_depth(self, tree):
        if tree.depth() > self.max_depth or tree.depth() < self.min_depth:
            return self._generate_random_tree()
        return tree
