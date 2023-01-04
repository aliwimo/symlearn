import numpy as np
from copy import deepcopy
from random import random
from datetime import datetime, timedelta
from symlearn.core.methods import Methods
from symlearn.core.functions import *
from symlearn.models.model import Model


class IPP(Model):

    def __init__(self,
                 pop_size=100,
                 donors_number=1,
                 receivers_number=1,
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

        super(IPP, self).__init__(max_evaluations,
                                  max_generations,
                                  max_time,
                                  verbose)
        self.pop_size = pop_size
        self.donors_number = donors_number
        self.receivers_number = receivers_number
        self.initial_min_depth = initial_min_depth
        self.initial_max_depth = initial_max_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.error_function = error_function
        self.expressions = expressions
        self.terminals = terminals
        self.target_error = target_error
        self.verbose = verbose
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

    # perform infection between two individuals
    def perform_infection(self, individual):
        return Methods.change_node(individual, self.expressions + self.terminals)

    # performing plasma tranfer from donor to receiver indvidual
    def perform_plasma_transfer(self, receiver, donor):
        return Methods.share(donor, deepcopy(receiver))

    # get lists of indexes of doreceivers_numbers and recievers
    def get_donors_and_receivers_indexes(self):
        donors = []
        receivers = []
        sorted_indexes = np.argsort(self.fitnesses)
        for i in range(self.donors_number):
            donors.append(sorted_indexes[i])
        for i in range(self.receivers_number):
            receivers.append(sorted_indexes[-1 - i])
        return donors, receivers

    def run(self):
        while not self.must_terminate():
            self.rank(is_reversed=False)

            # start of infection phase
            for index in range(self.pop_size):
                self.current_evaluation += 1
                random_index = np.random.randint(0, self.pop_size)
                while random_index == index:
                    random_index = np.random.randint(0, self.pop_size)
                current_individual = deepcopy(self.population[index])
                infected_individual = self.perform_infection(
                    current_individual)
                if infected_individual.depth() > self.max_depth or infected_individual.depth() < self.min_depth:
                    if random() > 0.5:
                        infected_individual = Methods.generate_individual(
                            'full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                    else:
                        infected_individual = Methods.generate_individual(
                            'grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                self.evalualte(index, infected_individual)
                if self.must_terminate():
                    break

            # start of plasma transfering phase
            # generating dose_control and treatment_control vectors
            dose_control = np.ones(self.receivers_number, int)
            treatment_control = np.ones(self.receivers_number, int)

            # get indexes of both of donors and receivers
            donors_indexes, receivers_indexes = self.get_donors_and_receivers_indexes()

            for index in range(self.receivers_number):
                receiver_index = receivers_indexes[index]
                random_donor_index = donors_indexes[int(
                    np.random.randint(0, self.donors_number))]
                current_receiver = self.population[receiver_index]
                random_donor = self.population[random_donor_index]
                while treatment_control[index] == 1:
                    if self.must_terminate():
                        break
                    self.current_evaluation += 1
                    treated_individual = self.perform_plasma_transfer(
                        current_receiver, random_donor)
                    treated_individual.update_fitness(
                        self.error_function, self.X, self.y)
                    if dose_control[index] == 1:
                        if treated_individual.fitness < self.fitnesses[random_donor_index]:
                            dose_control[index] += 1
                            self.population[receiver_index] = deepcopy(
                                treated_individual)
                            self.fitnesses[receiver_index] = treated_individual.fitness
                        else:
                            self.population[receiver_index] = deepcopy(
                                random_donor)
                            self.fitnesses[receiver_index] = self.fitnesses[random_donor_index]
                            treatment_control[index] = 0
                    else:
                        if treated_individual.fitness < self.fitnesses[receiver_index]:
                            self.population[receiver_index] = deepcopy(
                                treated_individual)
                            self.fitnesses[receiver_index] = treated_individual.fitness
                        else:
                            treatment_control[index] = 0

                    if self.population[receiver_index].fitness < self.model.fitness:
                        self.model = deepcopy(
                            self.population[receiver_index])
                if self.must_terminate():
                    break

            # start of donors updating phase
            for index in range(self.donors_number):
                if self.must_terminate():
                    break
                self.current_evaluation += 1
                donor_index = donors_indexes[index]
                if (self.current_evaluation / self.max_evaluations) > random():
                    # the same of update donor here!
                    temp = self.perform_infection(self.population[donor_index])
                else:
                    if random() > 0.5:
                        temp = Methods.generate_individual(
                            'full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                    else:
                        temp = Methods.generate_individual(
                            'grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                temp.update_fitness(self.error_function, self.X, self.y)
                self.population[donor_index] = deepcopy(temp)
                self.fitnesses[donor_index] = self.population[donor_index].fitness
                if self.population[donor_index].fitness < self.model.fitness:
                    self.model = deepcopy(
                        self.population[donor_index])
                if self.must_terminate():
                    break

            if not self.max_generations == -1:
                self.current_generation += 1
