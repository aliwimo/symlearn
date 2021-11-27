import numpy as np
from copy import deepcopy, copy
from random import randint, random
from parameters import Parameters
from functions import Functions

class Firefly:

    counter = 0
    gen = 0
    display_results = True

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.best = None
        self.best_gen = 0
        self.errors = [0] * Parameters.POP_SIZE
        self.population = Functions.generate_population()
        self.best_error = 10e6
        min_index = -1
        for index in range(Parameters.POP_SIZE):
            self.errors[index] = np.sum(np.abs(self.population[index].output(self.X) - self.Y))
            if self.errors[index] <= self.best_error: 
                min_index = index
                self.best_error = self.errors[index]
        self.best = deepcopy(self.population[min_index])

    def export_best(self):
        label = "Best founded at Generation: "
        label += str(self.best_gen)
        label += " and has error of: "
        label += str(round(self.best_error, 3))
        Functions.export_graph(self.best, "Best_mode", label)
        print(self.best.equation())


    def firefly_algorithm(self):
        while Firefly.counter < Parameters.MAX_EVAL:
            for i in range(Parameters.POP_SIZE):
                if Firefly.counter >= Parameters.MAX_EVAL: break
                for j in range(Parameters.POP_SIZE):
                    if Firefly.counter >= Parameters.MAX_EVAL: break
                    
                    if self.errors[i] >= self.errors[j]:
                        Firefly.counter += 1


                        temp = deepcopy(self.population[i])
                        temp = Functions.share(self.population[j], self.population[i])
                        temp_error = np.sum(np.abs(temp.output(self.X) - self.Y))
                        if temp_error < self.errors[i]:
                            self.population[i] = deepcopy(temp)
                            self.errors[i] = temp_error
                            if self.errors[i] < self.best_error:
                                self.best = deepcopy(self.population[i])
                                self.best_error = self.errors[i]
                                self.best_gen = Firefly.gen
                                if Firefly.display_results:
                                    print(f'Counter: {Firefly.counter}\t| Gen: {Firefly.gen}\t| Best: {self.best_error}')
                    
                    if self.best_error < 0.01: break
                if self.best_error < 0.01: break
            if self.best_error < 0.01: break
            Firefly.gen += 1

    def run(self):
        Firefly.display_results = True
        self.firefly_algorithm()
        print(self.best_error)
        self.export_best()
