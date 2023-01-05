import numpy as np
from datetime import datetime
from symlearn.core.methods import Methods
from symlearn.core.functions import *
from sklearn.metrics import r2_score
from copy import deepcopy


class Model:
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

    def generate_population(self):
        self.population = Methods.generate_population(
            pop_size=self.pop_size,
            initial_min_depth=self.initial_min_depth,
            initial_max_depth=self.initial_max_depth,
            expressions=self.expressions,
            terminals=self.terminals)

    def get_initial_statistics(self):
        self.fitnesses = [0] * self.pop_size
        min_error = 10e6
        min_index = -1
        for index in range(self.pop_size):
            self.population[index].update_fitness(
                self.error_function, self.X, self.y)
            self.fitnesses[index] = self.population[index].fitness
            if self.population[index].fitness <= min_error:
                min_index = index
        self.model = deepcopy(self.population[min_index])
        self.model.update_fitness(
            self.error_function, self.X, self.y)
        self.model.update_fitness(
            self.error_function, self.X, self.y)

    def rank(self, is_reversed=False):
        self.population, self.fitnesses = Methods.rank_trees(
            self.population, self.fitnesses, is_reversed)

    def evalualte(self, current, temp):
        temp.update_fitness(self.error_function, self.X, self.y)
        if temp.fitness < self.population[current].fitness:
            self.population[current] = deepcopy(temp)
            self.fitnesses[current] = self.population[current].fitness
            if self.population[current].fitness < self.model.fitness:
                self.model = deepcopy(self.population[current])
                if self.verbose:
                    print(
                        f'Evaluations: {self.current_evaluation} | Fitness: {self.model.fitness}')


    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def predict(self, X):
        return self.model.output(X)

    def must_terminate(self):
        terminate = False
        if self.max_time and datetime.now() > self.end_time:
            terminate = True
        elif self.max_evaluations > -1 and self.current_evaluation > self.max_evaluations:
            terminate = True
        elif self.max_generations > -1 and self.current_generation > self.max_generations:
            terminate = True
        elif self.model.fitness < self.target_error:
            terminate = True
        return terminate

    def export_best(self, export_path='images/', filename='Best'):
        if self.model:
            label = "Best error: "
            label += str(round(self.model.fitness, 3))
            Methods.export_graph(self.model, export_path + filename, label)
            print(self.model.equation())

    def test_model(self, X):
        zero_array = np.zeros(X.shape)
        x = np.vstack([zero_array, X])
        return self.model.output(x)[-1]
