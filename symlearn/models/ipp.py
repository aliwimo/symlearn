import numpy as np
from copy import deepcopy
from random import random
from datetime import datetime, timedelta
from symlearn.core.methods import Methods
from symlearn.core.operators import share, substitute
from symlearn.core.functions import *
from symlearn.models.model import Model


class IPP(Model):
    """
    A class for that represents Immune Plasma Programming (IPP) algorithm. This algorithm works by selecting the best
    tree at each generation and applying different operators to it to create new trees.
    """

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
        """
        Initializes the IPP algorithm.

        Args:
            donors_number (int): The number of donors.
            receivers_number (int): The number of receivers.
        """
        super(IPP, self).__init__(
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

        self.donors_number = donors_number
        self.receivers_number = receivers_number

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

    # perform infection between two individuals
    def _perform_infection(self, individual):
        """
        Perform infection between two individuals.

        Args:
            individual: An individual to infect.

        Returns:
            A modified version of `individual` with the infection.
        """
        return substitute(individual, self.expressions + self.terminals)

    # performing plasma tranfer from donor to receiver indvidual
    def _perform_plasma_transfer(self, receiver, donor):
        """
        Perform a plasma transfer from a donor individual to a receiver individual.

        Args:
            receiver: The individual receiving the plasma.
            donor: The individual donating the plasma.

        Returns:
            A modified version of `receiver` with the transferred plasma.
        """
        return share(donor, deepcopy(receiver))

    # get lists of indexes of donors and receivers
    def _get_donors_and_receivers_indexes(self):
        """
        Get lists of indexes of donors and receivers.

        Returns:
            A tuple containing two lists: the first is a list of indexes of donors,
            and the second is a list of indexes of receivers.
        """
        donors = []
        receivers = []
        sorted_indexes = np.argsort(self.fitnesses)
        for i in range(self.donors_number):
            donors.append(sorted_indexes[i])
        for i in range(self.receivers_number):
            receivers.append(sorted_indexes[-1 - i])
        return donors, receivers

    def _run(self):
        """
        Runs the Immune Plasma Programming (IPP) algorithm until termination conditions are met.

        Returns:
            None
        """
        while not self._should_terminate():
            self._rank(is_reversed=False)

            # start of infection phase
            for index in range(self.pop_size):
                self.current_evaluation += 1
                random_index = np.random.randint(0, self.pop_size)
                while random_index == index:
                    random_index = np.random.randint(0, self.pop_size)
                current_individual = deepcopy(self.population[index])
                infected_individual = self._perform_infection(
                    current_individual)
                if infected_individual.depth() > self.max_depth or infected_individual.depth() < self.min_depth:
                    if random() > 0.5:
                        infected_individual = Methods.generate_individual(
                            'full', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                    else:
                        infected_individual = Methods.generate_individual(
                            'grow', self.initial_min_depth, self.initial_max_depth, self.expressions, self.terminals)
                self._evaluate(index, infected_individual)
                if self._should_terminate():
                    break

            # start of plasma transferring phase
            # generating dose_control and treatment_control vectors
            dose_control = np.ones(self.receivers_number, int)
            treatment_control = np.ones(self.receivers_number, int)

            # get indexes of both of donors and receivers
            donors_indexes, receivers_indexes = self._get_donors_and_receivers_indexes()

            for index in range(self.receivers_number):
                receiver_index = receivers_indexes[index]
                random_donor_index = donors_indexes[int(
                    np.random.randint(0, self.donors_number))]
                current_receiver = self.population[receiver_index]
                random_donor = self.population[random_donor_index]
                while treatment_control[index] == 1:
                    if self._should_terminate():
                        break
                    self.current_evaluation += 1
                    treated_individual = self._perform_plasma_transfer(
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
                if self._should_terminate():
                    break

            # start of donors updating phase
            for index in range(self.donors_number):
                if self._should_terminate():
                    break
                self.current_evaluation += 1
                donor_index = donors_indexes[index]
                if (self.current_evaluation / self.max_evaluations) > random():
                    # the same of update donor here!
                    temp = self._perform_infection(self.population[donor_index])
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
                if self._should_terminate():
                    break

            if not self.max_generations == -1:
                self.current_generation += 1
