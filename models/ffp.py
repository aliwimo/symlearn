import numpy as np
from copy import deepcopy
from random import random
from datetime import datetime, timedelta
from core.methods import Methods
from core.functions import *
from models.model import Model


class FFP(Model):

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

        super(FFP, self).__init__(max_evaluations,
                                  max_generations,
                                  max_time,
                                  verbose)
        self.pop_size = pop_size
        self.initial_min_depth = initial_min_depth
        self.initial_max_depth = initial_max_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.error_function = error_function
        self.expressions = expressions
        self.terminals = terminals
        self.target_error = target_error
        self.population = None
        self.fitnesses = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.max_time:
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(seconds=self.max_time)
        self.generate_population()
        self.get_initial_statistics()
        self.run()
        if self.verbose:
            if self.max_time:
                print(f'Total time: {datetime.now() - self.start_time}')
            print(f'Evaluations: {self.current_evaluation}')

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

    def attract(self, i, j):
        return Methods.share(self.population[j], deepcopy(self.population[i]))

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

    def run(self):
        while not self.must_terminate():
            self.rank(is_reversed=False)
            for i in range(self.pop_size):
                if self.must_terminate():
                    break
                for j in range(self.pop_size):
                    if self.must_terminate():
                        break
                    if self.population[i].fitness >= self.population[j].fitness:
                        temp = self.attract(i, j)
                        if temp.depth() > self.max_depth or temp.depth() < self.min_depth:
                            if random() > 0.5:
                                temp = Methods.generate_individual(
                                    'full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                            else:
                                temp = Methods.generate_individual(
                                    'grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                        self.evalualte(i, temp)
                        self.current_evaluation += 1

                    if self.must_terminate():
                        break
                if self.must_terminate():
                    break
            if self.must_terminate():
                break
            # increase generation counter
            if not self.max_generations == -1:
                self.current_generation += 1
