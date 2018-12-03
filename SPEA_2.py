"""SPEA_2 algorithm"""

import numpy as np
from Abstract_Moo import *
import itertools as itr
from moo_sorting_functions import *
import pandas as pd
import time


class SPEA_2(AbstractMOEA):
    """Represents the SPEA_2 algorithm"""

    def __init__(self, objectives, bounds, population_size=50, archive_size=50, cross_prob=0.9, cross_dist=15,
                 mut_prob=0.01, mut_dist=20, iterations=20):
        super().__init__(objectives, bounds, cross_prob, cross_dist, mut_prob, mut_dist, iterations)
        self.population_size = population_size
        self.archive_size = archive_size
        self.population = []
        self.archive = []
        self.next_archive = []
        self.joint = []
        self.distance_graph = dict()

    def run(self):
        self.population = self.initialise_population(self.population_size, PopIndividual)
        self.archive = []
        for _ in range(self.iterations):
            self.joint = self.population + self.archive
            self.fitness_assignment()  # Assigns fitness values to all population members
            t0 = time.perf_counter()
            self.environmental_selection()  # create new archive from current archive and population
            t1 = time.perf_counter()
            print("Enviromental Selection Time:", t1 - t0)
            next_population = self.tournament_selection(self.archive, self.population_size)
            self.crossover_step_SBX(next_population)
            self.mutation_step_polynomial(next_population)
            self.population = next_population
        return self.archive

    def fitness_assignment(self):
        k = int(np.sqrt(self.population_size + self.archive_size)) - 1
        strength_dict = dict()
        for i, individual1 in enumerate(self.joint):
            distances = np.zeros(len(self.joint) - 1)
            for j, individual2 in itr.filterfalse(lambda x: x[0] == i, enumerate(self.joint)):
                if j > i:
                    j -= 1
                if individual1.dominates(individual2):
                    strength_dict[individual1] = strength_dict.get(individual1, 0) + 1
                    individual2.dominating_set.append(individual1)
                distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
                distances[j] = distance
            individual1.fitness_distances = distances
        for individual1 in self.joint:
            raw_fitness = 0
            for individual2 in individual1.dominating_set:
                raw_fitness += strength_dict.get(individual2, 0)
            individual1.fitness_distances.partition(k)
            density = individual1.fitness_distances[k]
            density = 1 / (density + 2)
            individual1.fitness = raw_fitness + density

    def environmental_selection(self):
        self.next_archive = self.get_non_dominated()
        distances_initialised = False
        distances = None
        while len(self.next_archive) > self.archive_size:
            if not distances_initialised:
                self.distance_assignment()
                distances_initialised = True
            self.distance_sort()
            self.truncate(distances)
        while len(self.next_archive) < self.archive_size:
            print(len(self.next_archive))
            k = self.archive_size - len(self.next_archive)
            a = np.partition(self.joint, k - 1)[:k]
            self.next_archive += list(a)
        self.archive = self.next_archive

    def get_non_dominated(self):
        mark = 0
        for i in range(len(self.joint)):
            if self.joint[i].fitness <= 1:
                if mark < i:
                    self.joint[i], self.joint[mark] = self.joint[mark], self.joint[i]
                mark += 1
        next_archive, self.joint = self.joint[:mark], self.joint[mark:]
        return next_archive

    def distance_assignment(self):
        for individual1, individual2 in itr.combinations(self.next_archive, 2):
            distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
            individual1.connect(individual2, distance)
        # distances = pd.DataFrame(columns=self.next_archive, index=self.next_archive)
        # for individual1, individual2 in itr.combinations(self.next_archive, 2):
        #     distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
        #     distances[individual1][individual2] = distance
        #     distances[individual2][individual1] = distance
        # return distances

    def truncate(self):
        for i, individual1 in enumerate(self.next_archive):
            min_individual = True
            j = 0
            while min_individual and j < len(self.next_archive):
                min_individual = individual1.less_than(self.next_archive[j])
                j += 1
            if min_individual:
                self.next_archive.pop(i)
                i = len(self.next_archive)
                individual1.delete()
            i += 1
        # i = 0
        # while i < len(self.next_archive):
        #     individual1 = self.next_archive[i]
        #     min_individual = True
        #     j = 0
        #     while min_individual and j < len(self.next_archive):
        #         min_individual = individual1.less_than(self.next_archive[j])
        #         j += 1
        #     if min_individual:
        #         distances.drop(index=individual1, columns=individual1)
        #         self.next_archive.pop(i)
        #         i = len(self.next_archive)
        #     i += 1

    def distance_sort(self):
        for individual in self.next_archive:
            individual.sort()
        # for individual1 in self.next_archive:
        #     individual1.distances = np.array(distances[individual1].dropna())
        #     individual1.distances.sort()


class PopIndividual(AbstractPopIndividual):
    """PopIndividual for SPEA algoritm"""

    def __init__(self, values, objectives, objective_values=None):
        super().__init__(values, objectives, objective_values=objective_values)
        self.dominating_set = []
        self.fitness_distances = None
        self.keys = []
        self.distances = dict()

    def __str__(self):
        return "values: {}, objectives: {}".format(self.values, self.objective_values)

    def __repr__(self):
        return str(self)

    def connect(self, other, distance):
        other.keys.append(self)
        other.distances[self] = distance
        self.keys.append(other)
        self.distances[other] = distance

    def disconnect(self, other):
        self.distances.pop(other)
        other.distances.pop(self)

    def delete(self):
        for other in self.keys:
            self.disconnect(other)

    def sort(self):
        self.keys.sort(key=lambda x: self.distances.get(x, np.inf))
        i = -1
        while abs(i) <= len(self.keys) and self.distances.get(self.keys[i]) is None:
            self.keys.pop(i)
            i -= 1

    def less_than(self, other):
        eq = True
        lt = False
        i = 0
        while i < len(self.keys) and not lt and eq:
            eq = self.distances[self.keys[i]] <= other.distances[other.keys[i]]
            lt = self.distances[self.keys[i]] < other.distances[other.keys[i]]
            i += 1
        return eq or lt


class Tests:

    @staticmethod
    def test_distance_less_than():
        objectives = [lambda x: x[0]]
        a = PopIndividual([1], objectives)
        b = PopIndividual([2], objectives)
        c = PopIndividual([3], objectives)
        d = PopIndividual([4], objectives)
        a.connect(c, 1)
        a.connect(d, 3)
        a.connect(b, 5)
        b.connect(c, 4)
        b.connect(d, 7)
        d.connect(c, 2)
        a.sort()
        b.sort()
        c.sort()
        d.sort()
        assert c.less_than(a)
        assert a.less_than(c) is False
        assert c.less_than(b)
        assert b.less_than(c) is False
        assert c.less_than(d)
        assert d.less_than(c) is False
        a.delete()
        b.sort()
        c.sort()
        d.sort()
        b.delete()
        c.sort()
        d.sort()
        assert c.less_than(d)
        assert d.less_than(c)

    @staticmethod
    def test_graph_operations():
        objectives = [lambda x: x[0]]
        a = PopIndividual([1], objectives)
        b = PopIndividual([2], objectives)
        c = PopIndividual([3], objectives)
        d = PopIndividual([4], objectives)
        a.connect(c, 1)
        a.connect(d, 3)
        a.connect(b, 5)
        b.connect(c, 4)
        b.connect(d, 7)
        d.connect(c, 2)
        a.sort()
        b.sort()
        c.sort()
        d.sort()
        assert a.keys == [c, d, b]
        assert b.keys == [c, a, d]
        assert c.keys == [a, d, b]
        assert d.keys == [c, a, b]
        c.delete()
        a.sort()
        b.sort()
        d.sort()
        assert a.keys == [d, b]
        assert b.keys == [a, d]
        assert d.keys == [a, b]

    @staticmethod
    def test_fitness_assignment():
        objectives = [lambda x: x[0], lambda x: 1/x[0] + x[1]]
        bounds = [(-1, 1), (-1, 1)]
        moo = SPEA_2(objectives, bounds, population_size=9, archive_size=0)
        for i in range(1, 4):
            for j in range(1, 4):
                moo.population.append(PopIndividual([j, i], objectives))
        moo.joint = moo.population + moo.archive
        moo.fitness_assignment()
        pop = moo.population
        assert np.isclose(pop[0].fitness, 0.320715)
        assert np.isclose(pop[1].fitness, 0.320715)
        assert np.isclose(pop[2].fitness, 0.282758)
        assert np.isclose(pop[3].fitness, 6.320715)
        assert np.isclose(pop[4].fitness, 10.33181)
        assert np.isclose(pop[5].fitness, 12.33181)
        assert np.isclose(pop[6].fitness, 9.262966)
        assert np.isclose(pop[7].fitness, 15.32071)
        assert np.isclose(pop[8].fitness, 18.30287)

    @staticmethod
    def test_truncate():
        objectives = [lambda x: x[0]]
        bounds = []
        moo = SPEA_2(objectives, bounds)
        a = PopIndividual([1], objectives)
        b = PopIndividual([2], objectives)
        c = PopIndividual([3], objectives)
        d = PopIndividual([4], objectives)
        a.connect(c, 1)
        a.connect(d, 3)
        a.connect(b, 5)
        b.connect(c, 4)
        b.connect(d, 7)
        d.connect(c, 2)
        a.sort()
        b.sort()
        c.sort()
        d.sort()
        moo.next_archive = [a, b, c, d]
        moo.truncate()
        assert moo.next_archive == [a, b, d]
        a.sort()
        b.sort()
        d.sort()
        moo.truncate()
        assert moo.next_archive == [b, d]



def main():
    Tests.test_distance_less_than()
    Tests.test_graph_operations()
    Tests.test_fitness_assignment()
    Tests.test_truncate()

main()

