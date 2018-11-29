"""SPEA_2 algorithm"""

import numpy as np
from Abstract_Moo import *
import itertools as itr
from moo_sorting_functions import *
import pandas as pd

class SPEA_2(AbstractMOEA):
    """Represents the SPEA_2 algorithm"""

    def __init__(self, objectives, bounds, population_size=80, archive_size=20, cross_prob=0.8, cross_dist=20,
                 mut_prob=0.01, mut_dist=20, iterations=20):
        super().__init__(objectives, bounds, cross_prob, cross_dist, mut_prob, mut_dist, iterations)
        self.population_size = population_size
        self.archive_size = archive_size
        self.k = np.sqrt(population_size + archive_size)
        self.population = None
        self.archive = None

    def run(self):
        self.population = self.initialise_population(self.population_size, PopIndividual)
        self.archive = []
        for _ in range(self.iterations):
            self.fitness_assignment()  # Assigns fitness values to all population members
            self.environmental_selection()  # create new archive from current archive and population
            next_population = self.tournament_selection(self.archive, self.population_size)
            self.crossover_step_SBX(next_population)
            self.mutation_step_polynomial(next_population)
            self.population = next_population
        return self.archive

    def fitness_assignment(self):
        strength_dict = dict()
        for individual1, individual2 in itr.combinations(self.population + self.archive, 2):
            if individual1.dominates(individual2):
                strength_dict[individual1] = strength_dict.get(individual1, 0), + 1
                individual1.dominated_set.append(individual2)
            elif individual2.dominates(individual1):
                strength_dict[individual1] = strength_dict.get(individual1, 0), + 1
                individual1.dominated_set.append(individual2)
            distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
            individual1.distances[individual2] = distance
            individual2.distances[individual1] = distance
        for individual1 in self.population + self.archive:
            raw_fitness = 0
            for individual2 in individual1.dominated_set:
                raw_fitness += strength_dict.get(individual2, 0)
            density = individual1.distances.partition(self.k)[self.k]
            density = 1 / (density + 2)
            individual1.fitness = raw_fitness + density

    def environmental_selection(self):
        next_archive = []
        for i in range(self.population_size):
            if self.population[i].fitness <= 1:
                next_archive.append(self.population.pop(i))
        for i in range(self.archive_size):
            if self.archive[i] <= 1:
                next_archive.append(self.archive.pop(i))
        archive_completed = False
        distances_initialised = False
        while not archive_completed:
            if len(self.archive) > self.archive_size:
                if not distances_initialised:
                    distances = pd.DataFrame(columns=self.archive, index=self.archive)
                    for individual1, individual2 in itr.combinations(self.archive, 2):
                        distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
                        distances[individual1][individual2] = distance
                        distances[individual2][individual1] = distance
                    for individual1 in self.archive:
                        individual1.distances = distances[individual1].sort_values()
            elif len(self.archive) < self.archive_size:
                next_archive += np.partition(self.archive + self.population, self.archive_size - len(next_archive))
            else:
                archive_completed = False


class PopIndividual(AbstractPopIndividual):
    """PopIndividual for SPEA algoritm"""

    def __init__(self, values, objectives, population_size):
        super().__init__(values, objectives)
        self.dominated_set = []
        self.distances = np.zeros(population_size - 1)

    def __str__(self):
        return "values: {}, objectives: {}".format(self.values, self.objective_values)

    def __repr__(self):
        return str(self)
