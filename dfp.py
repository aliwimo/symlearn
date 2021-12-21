import numpy as np
from copy import deepcopy
from random import randint, random
from tree import Tree
from parameters import Parameters
from methods import Methods

class DFP:

    def __init__(self,
                pop_size=100,
                alpha=0.1, 
                beta=0.5, 
                gamma=1.5,
                max_evaluations=10000,
                max_generations=-1,
                initial_min_depth=0,
                initial_max_depth=6,
                max_depth=15,
                target_error=0.0, 
                verbose=True
                ):

        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.initial_min_depth = initial_min_depth
        self.initial_max_depth = initial_max_depth
        self.max_depth = max_depth
        self.target_error = target_error
        self.verbose = verbose
        
        self.current_evaluation = 0
        self.current_generation = 0

        self.best_individual = None
        self.population = None
        self.errors = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.generate_population()
        self.get_initial_statistics()
        self.run()
        print(self.best_individual.error)
        return self.best_individual.calc_tree(X)

    def predict(self, X):
        return self.best_individual.calc_tree(X)

    def score(self, y_test, y_predict):
        return abs(100 - np.sum(np.abs(y_test - y_predict))) / 100

    def generate_population(self):
        self.population = Methods.init_trees(self.pop_size, self.initial_min_depth, self.initial_max_depth)

    def get_initial_statistics(self):
        self.errors = [0] * self.pop_size
        min_error = 10e6
        min_index = -1
        for index in range(self.pop_size):
            self.population[index].update_error(self.X, self.y)
            self.errors[index] = self.population[index].error
            if self.population[index].error <= min_error: 
                min_index = index
        self.best_individual = deepcopy(self.population[min_index])
        self.best_individual.update_error(self.X, self.y)

    def must_terminate(self):
        terminate = False
        if self.max_evaluations > -1 and self.current_evaluation > self.max_evaluations:
            terminate = True
        elif self.max_generations >-1 and self.current_generation > self.max_generations:
            terminate = True
        elif self.best_individual.error < self.target_error:
            terminate = True
        return terminate

    def rank(self, is_reversed=False):
        self.population, self.errors = Methods.rank_trees(self.population, self.errors, is_reversed)

    def export_best(self):
        if self.best_individual:
            label = "Best error: "
            label += str(round(self.best_individual.error, 3))
            self.best_individual.draw_tree("best_model", label)
            print(self.best_individual.tree_equation())

    def attract(self, i, j):
        distance = np.abs(self.population[i].error - self.population[j].error)
        temp = deepcopy(self.population[i])
        if distance > self.gamma:
            for x in range(3):
                Methods.share(self.population[j], temp)
        elif self.gamma >= distance > self.beta:
            for x in range(2):
                Methods.share(self.population[j], temp)
        else:
            Methods.share(self.population[j], temp)
        return temp

    def evalualte(self, current, temp):
        if temp.error < self.population[current].error:
            self.population[current] = deepcopy(temp)
            self.errors[current] = self.population[current].error
            if self.population[current].error < self.best_individual.error:
                self.best_individual = deepcopy(self.population[current])
                if self.verbose:
                    print(f'Evaluations: {self.current_evaluation}\t| Gen: {self.current_generation}\t| Error: {self.best_individual.error}')
    
    # standard firefly programming method (FP)
    def run(self):
        while not self.must_terminate():
            self.rank(is_reversed=False)

            for i in range(self.pop_size):
                if self.must_terminate(): break

                for j in range(self.pop_size):
                    if self.must_terminate(): break

                    if self.population[i].error >= self.population[j].error:

                        self.current_evaluation += 1
                        
                        temp = self.attract(i, j)
                        temp = Methods.check_depth(temp, self.initial_min_depth, self.initial_max_depth, self.max_depth)
                        temp.update_error(self.X, self.y)
                        self.evalualte(i, temp)

                    if self.must_terminate(): break
                if self.must_terminate(): break
            if self.must_terminate(): break
            
            # increase generation counter
            if not self.max_generations == -1: self.current_generation += 1
