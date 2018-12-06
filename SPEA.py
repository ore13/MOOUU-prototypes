"""
Strength pareto evolutionary algorithim - Zitzler
Benchmark testing for SAM MOOUU using differential evolution
GNS.cri
Otis Rea
"""

import numpy as np
from Abstract_Moo import AbstractPopIndividual, AbstractMOEA


class SPEA(AbstractMOEA):

    def __init__(self, objectives, bounds, constraints=None, population_size=80, archive_size=20, cross_prob=0.9,
                 cross_dist=15, mut_prob=0.01, mut_dist=20, iterations=20):
        super().__init__(objectives, bounds, constraints, cross_prob, cross_dist, mut_prob, mut_dist, iterations)
        self.population_size = population_size
        self.archive_size = archive_size
        self.population = []
        self.archive = []

    def run(self):
        self.population = self.initialise_population(self.population_size, PopIndividual)
        for _ in range(self.iterations):
            temporary_archive = self.fast_non_dominated_front(self.population)
            for individual in temporary_archive:
                self.archive.append(individual.clone())
            self.archive = self.fast_non_dominated_front(self.archive, covers=True)
            if len(self.archive) > self.archive_size:
                self.reduce_archive()
            self.fitness_assignment()
            self.population = self.tournament_selection(self.population + self.archive, self.population_size)
            self.crossover_step_SBX(self.population)
            self.mutation_step_polynomial(self.population)
            self.reset_population(self.archive)
        return self.archive

    def __str__(self):
        return "SPEA"

    @staticmethod
    def cluster_distance(cluster1, cluster2, distance_dict):
        distance = 0
        for pop1 in cluster1:
            for pop2 in cluster2:
                distance += distance_dict.get((pop1, pop2), distance_dict.get((pop2, pop1)))
        distance = 1 / (len(cluster1) * len(cluster2)) * distance
        return distance

    @staticmethod
    def cluster_centroid(cluster, distance_dict):
        centroid = None
        min_distance = np.inf
        for i, individual1 in enumerate(cluster):
            distance = 0
            for individual2 in cluster[:i] + cluster[i+1:]:
                alternative = distance_dict.get((individual2, individual1))
                distance += distance_dict.get((individual1, individual2), alternative)
            if distance < min_distance:
                min_distance = distance
                centroid = individual1
        return centroid

    def reduce_archive(self):
        """Clustering algorithms are nightmarishly inefficient"""
        distance_dict = dict()
        clusters = []
        for i, pop1 in enumerate(self.archive):
            clusters.append([pop1])
            for j in range(i + 1, len(self.archive)):
                pop2 = self.archive[j]
                distance_dict[(pop1, pop2)] = np.linalg.norm(pop1.objective_values - pop2.objective_values, 2)
        while len(clusters) > self.archive_size:
            min_distance = np.inf
            min_clusters = (np.nan, np.nan)
            for i, cluster1 in enumerate(clusters):
                for j in range(i + 1, len(clusters)):
                    cluster2 = clusters[j]
                    distance = SPEA.cluster_distance(cluster1, cluster2, distance_dict)
                    if distance < min_distance:
                        min_distance = distance
                        min_clusters = (i, j)
            cluster1 = clusters.pop(max(min_clusters))
            cluster2 = clusters.pop(min(min_clusters))
            clusters.append(cluster1 + cluster2)
        self.archive = []
        for cluster in clusters:
            self.archive.append(self.cluster_centroid(cluster, distance_dict))
        # TODO: again objective_values is being used. solutions in a constrained problem may not have objective_values

    def fitness_assignment(self):
        for individual1 in self.archive:
            n = 0
            for individual2 in self.population:
                if self.covers(individual1, individual2):
                    n += 1
                    individual2.covered_by_set.append(individual1)
            individual1.fitness = n / (self.population_size + 1)
        for individual1 in self.population:
            fitness = 1
            for individual2 in individual1.covered_by_set:
                fitness += individual2.fitness
            individual1.fitness = fitness


class PopIndividual(AbstractPopIndividual):
    """represents an individual in a population for SPEA"""

    def __init__(self, d_vars, objectives, constraints=None, objective_values=None, constrained_values=None):
        super().__init__(d_vars, objectives, constraints, objective_values, constrained_values)
        self.covered_by_set = []

    def __str__(self):
        return "d_vars: {}, objectives: {}".format(self.d_vars, self.objective_values)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        """defining __lt__ for Kung's algorithim sorting procedure"""
        if self.violates and other.violates:
            result = self.total_constraint_violation <= other.total_constraint_violation
        elif self.violates:
            result = False
        elif other.violates:
            result = True
        else:
            result = self.objective_values[0] <= other.objective_value[0]

    def clear(self):
        self.covered_by_set = []
        self.fitness = 0
