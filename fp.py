import numpy as np
from copy import deepcopy
from random import random
from node import Node
from parameters import Parameters
from methods import Methods
from functions import *
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

class FP:

    def __init__(self,
                pop_size=100,
                alpha=0.1, 
                beta=0.5, 
                gamma=1.5,
                max_evaluations=10000,
                max_generations=-1,
                max_time=None,
                initial_min_depth=0,
                initial_max_depth=6,
                max_depth=15,
                error_function=None,
                expressions=[Add, Sub, Mul],
                terminals=[Variable, Constant],
                target_error=0.0, 
                verbose=True
                ):

        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.max_time = max_time
        self.initial_min_depth = initial_min_depth
        self.initial_max_depth = initial_max_depth
        self.max_depth = max_depth
        self.error_function = error_function
        self.expressions = expressions
        self.terminals = terminals
        self.target_error = target_error
        self.verbose = verbose
        self.current_evaluation = 0
        self.current_generation = 0
        self.best_individual = None
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
            print(f'Total time: {datetime.now() - self.start_time}')
            print(f'Evaluations: {self.current_evaluation}')

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def predict(self, X):
        return self.best_individual.output(X)

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
            self.population[index].update_fitness(self.error_function, self.X, self.y)
            self.fitnesses[index] = self.population[index].fitness
            if self.population[index].fitness <= min_error: min_index = index
        self.best_individual = deepcopy(self.population[min_index])
        self.best_individual.update_fitness(self.error_function, self.X, self.y)
        self.best_individual.update_fitness(self.error_function, self.X, self.y)

    def must_terminate(self):
        terminate = False
        if self.max_time and datetime.now() > self.end_time: 
            terminate = True
        elif self.max_evaluations > -1 and self.current_evaluation > self.max_evaluations:
            terminate = True
        elif self.max_generations >-1 and self.current_generation > self.max_generations:
            terminate = True
        elif self.best_individual.fitness < self.target_error:
            terminate = True
        return terminate

    def rank(self, is_reversed=False):
        self.population, self.fitnesses = Methods.rank_trees(self.population, self.fitnesses, is_reversed)
    
    def export_best(self, filename='Best'):
        if self.best_individual:
            label = "Best error: "
            label += str(round(self.best_individual.fitness, 3))
            Methods.export_graph(self.best_individual, 'images\\' + filename, label)
            print(self.best_individual.equation())

    def attract(self, i, j):
        return Methods.share(self.population[j], deepcopy(self.population[i]))

    def evalualte(self, current, temp):
        temp.update_fitness(self.error_function, self.X, self.y)
        if temp.fitness < self.population[current].fitness:
            self.population[current] = deepcopy(temp)
            self.fitnesses[current] = self.population[current].fitness
            if self.population[current].fitness < self.best_individual.fitness:
                self.best_individual = deepcopy(self.population[current])
                if self.verbose:
                    print(f'Evaluations: {self.current_evaluation} | Fitness: {self.best_individual.fitness}')
    
    # standard firefly programming method (FP)
    def run(self):
        while not self.must_terminate():
            self.rank(is_reversed=False)
            for i in range(self.pop_size):
                if self.must_terminate(): break
                for j in range(self.pop_size):
                    if self.must_terminate(): break
                    if self.population[i].fitness >= self.population[j].fitness:
                        temp = self.attract(i, j)
                        if temp.depth() > self.max_depth:
                            if random() > 0.5:
                                temp = Methods.generate_individual('full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                            else:
                                temp = Methods.generate_individual('grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                        self.evalualte(i, temp)
                        self.current_evaluation += 1

                    if self.must_terminate(): break
                if self.must_terminate(): break
            if self.must_terminate(): break
            # increase generation counter
            if not self.max_generations == -1: self.current_generation += 1

