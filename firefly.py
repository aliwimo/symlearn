import numpy as np
from copy import deepcopy
from random import randint, random
from plot import plot2D, plot3D, prepare_2Dplots, prepare_3Dplots
from tree import Tree
from parameters import Par
from functions import Func

class Firefly:

    counter = 0
    gen = 0
    display_results = True

    def __init__(self, alpha, beta, gamma):
        self.target_ds = Func.generate_dataset()
        self.best = None
        self.best_gen = 0
        self.errors = [0] * Par.POP_SIZE
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = Func.init_trees()
        min_error = 10e6
        min_index = -1
        for index in range(len(self.population)):
            self.population[index].update_error()
            self.errors[index] = self.population[index].error
            if self.population[index].error <= min_error: 
                min_index = index
        self.best = deepcopy(self.population[min_index])
        self.best.update_error()
        # self.simplify_trees()
        
    def simplify_trees(self):
        for index in range(len(self.population)):
            self.population[index].simplify_tree()

    def rank(self, is_reversed=False):
        self.population, self.errors = Func.rank_trees(self.population, self.errors, is_reversed)
    
    def plot(self):
        if Par.VAR_NUM == 1: ax, line = prepare_2Dplots(self.target_ds)
        else: ax, line = prepare_3Dplots(self.target_ds)
        computed_ds = Func.computed_dataset(self.best)
        if Par.VAR_NUM == 1: plot2D(ax, line, computed_ds, last=True)
        else: plot3D(ax, line, computed_ds, last=True)

    def export_best(self):
        label = "Best founded at Generation: "
        label += str(self.best_gen)
        label += " and has error of: "
        label += str(round(self.best.error, 3))
        self.best.draw_tree("best_model", label)
        print(self.best.tree_equation())

    def attract(self, i, j):
        distance = np.abs(self.population[i].error - self.population[j].error)
        temp = deepcopy(self.population[i])
        Func.share(self.population[j], temp)
        # if distance > self.gamma:
        #     for _ in range(3):
        #         Func.share(self.population[j], temp)
        #     temp.change_node()
        # elif self.beta < distance <= self.gamma:
        #     for _ in range(2):
        #         Func.share(self.population[j], temp)
        #     temp.change_node()
        # elif self.alpha < distance <= self.beta:
        #     for _ in range(2):
        #         Func.share(self.population[j], temp)
        # else:
        #     Func.share(self.population[j], temp)
        return temp

    def evalualte(self, current, temp):
        if temp.error < self.population[current].error:
            self.population[current] = deepcopy(temp)
            self.errors[current] = self.population[current].error
            if self.population[current].error < self.best.error:
                self.best = deepcopy(self.population[current])
                self.best_gen = Firefly.gen
                if Firefly.display_results:
                    print(f'Counter: {Firefly.counter}\t| Gen: {Firefly.gen}\t| Best: {self.best.error}')
    
    def firefly_algorithm(self):
        while Firefly.counter < Par.MAX_EVAL:
            self.rank(is_reversed=False)
            for i in range(Par.POP_SIZE):
                if Firefly.counter >= Par.MAX_EVAL: break
                for j in range(Par.POP_SIZE):
                    if Firefly.counter >= Par.MAX_EVAL: break

                    if self.population[i].error >= self.population[j].error:
                        Firefly.counter += 1
                        # Func.progress_bar(Firefly.counter, Firefly.gen, self.best.error, Par.MAX_EVAL, self.alpha, length=50)           
                        
                        temp = self.attract(i, j)
                        temp = Func.check_depth(temp)
                        is_different = Func.control_difference(self.population, temp)
                        while not is_different:
                            method = 'grow' if randint(1, 2) == 1 else 'full'
                            temp.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
                            is_different = Func.control_difference(self.population, temp)
                        temp.update_error()
                        self.evalualte(i, temp)

                    if self.best.error < 0.01: break
                if self.best.error < 0.01: break
            if self.best.error < 0.01: break
            Firefly.gen += 1



    def DFP(self):
        while Firefly.counter < Par.MAX_EVAL:
            self.rank(is_reversed=False)
            for i in range(Par.POP_SIZE):
                if Firefly.counter >= Par.MAX_EVAL: break
                for j in range(Par.POP_SIZE):
                    if Firefly.counter >= Par.MAX_EVAL: break

                    if self.population[i].error >= self.population[j].error:
                        difference = abs(self.population[i].error - self.population[j].error)
                        Firefly.counter += 1
                        # Func.progress_bar(Firefly.counter, Firefly.gen, self.best.error, Par.MAX_EVAL, self.alpha, length=50)           
                        
                        # if difference >= self.gamma:
                        #     for _ in range(3):
                        #         temp = self.attract(i, j)
                        # elif self.gamma > difference >= self.beta:
                        #     for _ in range(2):
                        #         temp = self.attract(i, j)
                        # elif self.beta > difference >= self.alpha:
                        #     temp = self.attract(i, j)
                        # else: temp =  



                        temp = self.attract(i, j)
                        temp = Func.check_depth(temp)
                        is_different = Func.control_difference(self.population, temp)
                        while not is_different:
                            method = 'grow' if randint(1, 2) == 1 else 'full'
                            temp.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
                            is_different = Func.control_difference(self.population, temp)
                        temp.update_error()
                        self.evalualte(i, temp)

                    if self.best.error < 0.01: break
                if self.best.error < 0.01: break
            if self.best.error < 0.01: break
            Firefly.gen += 1

    def run(self):
        Firefly.display_results = True
        # self.firefly_algorithm()
        self.DFP()
        print(self.best.error)
        # self.plot()
        self.export_best()
