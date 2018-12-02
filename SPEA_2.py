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
        self.population = []
        self.archive = []
        self.joint = []

    def run(self):
        self.population = self.initialise_population(self.population_size, PopIndividual)
        self.archive = []
        for _ in range(self.iterations):
            self.joint = self.population + self.archive
            self.fitness_assignment()  # Assigns fitness values to all population members
            self.environmental_selection()  # create new archive from current archive and population
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
            individual1.distances = distances
        for individual1 in self.joint:
            raw_fitness = 0
            for individual2 in individual1.dominating_set:
                raw_fitness += strength_dict.get(individual2, 0)
            density = np.partition(individual1.distances, k)[k]
            density = 1 / (density + 2)
            individual1.fitness = raw_fitness + density

    def environmental_selection(self):
        next_archive = []
        mark = 0
        for i in range(len(self.joint)):
            if self.joint[i].fitness <= 1:
                if mark < i:
                    self.joint[i], self.joint[mark] = self.joint[mark], self.joint[i]
                mark += 1
        next_archive, self.joint = self.joint[:mark], self.joint[mark:]
        archive_completed = False
        distances_initialised = False
        print(len(next_archive))
        while not archive_completed:
            if len(next_archive) > self.archive_size:
                if not distances_initialised:
                    distances = pd.DataFrame(columns=next_archive, index=next_archive)
                    for individual1, individual2 in itr.combinations(next_archive, 2):
                        distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
                        distances[individual1][individual2] = distance
                        distances[individual2][individual1] = distance
                    distances_initialised = True
                for individual1 in next_archive:
                    individual1.distances = distances[individual1].sort_values().dropna()
                i = 0
                while i < len(next_archive):
                    individual1 = next_archive[i]
                    min_individual = True
                    j = 0
                    while min_individual and j < len(next_archive):
                        individual2 = next_archive[j]
                        min_individual = individual1.less_than(individual2)
                        j += 1
                    if min_individual:
                        distances.drop(index=individual1, columns=individual1)
                        del next_archive[i]
                        i = len(next_archive)
                    i += 1
            elif len(next_archive) < self.archive_size:
                print(len(next_archive))
                k = self.archive_size - len(next_archive)
                a = np.partition(self.joint, k - 1)[:k]
                next_archive += list(a)
            else:
                archive_completed = True
        self.archive = next_archive


class PopIndividual(AbstractPopIndividual):
    """PopIndividual for SPEA algoritm"""

    def __init__(self, values, objectives, objective_values=None):
        super().__init__(values, objectives, objective_values=objective_values)
        self.dominating_set = []
        self.distances = None

    def __str__(self):
        return "values: {}, objectives: {}".format(self.values, self.objective_values)

    def __repr__(self):
        return str(self)

    def less_than(self, other):
        eq = True
        lt = False
        i = 0
        while i < len(self.distances) and not lt and eq:
            eq = bool(self.distances[i] <= other.distances[i])
            lt = bool(self.distances[i] < other.distances[i])
            i += 1
        return eq or lt


class Tests:

    @staticmethod
    def test_distance_less_than():
        objectives = []
        values = []
        a = PopIndividual(values, objectives)
        b = PopIndividual(values, objectives)
        a.distances = np.array([1., 2., 3.])
        b.distances = np.array([2., 2., 3.])
        assert a.less_than(b)
        assert (b.less_than(a) is False)
        b.distances = np.array([1., 2., 4.])
        assert a.less_than(b)
        assert b.less_than(a) is False
        b.distances = np.array([1., 2., 3.])
        assert b.less_than(a)
        assert a.less_than(b)
        b.distances = np.array([1., 1., 4.])
        assert a.less_than(b) is False
        assert b.less_than(a)
        assert a.less_than(a)


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
    def test_environmental_selection():
        objectives = [lambda x: x]
        bounds = []
        moo = SPEA_2(objectives, bounds, archive_size=3, population_size=6)
        a = PopIndividual([1], objectives)
        a.fitness = 0.1
        b = PopIndividual([2], objectives)
        b.fitness = 0.9
        c = PopIndividual([3], objectives)
        c.fitness = 1.0
        d = PopIndividual([4], objectives)
        d.fitness = 1.01
        e = PopIndividual([5], objectives)
        e.fitness = 1.5
        f = PopIndividual([6], objectives)
        f.fitness = 10
        pop = [a, b, c, d, e, f]
        moo.population = pop
        moo.joint = moo.population + moo.archive
        moo.environmental_selection()
        archive = moo.archive
        assert archive == [a, b, c]
        moo.archive = []
        c.fitness = 1.2
        moo.population = [a, b, c, d, e, f]
        moo.joint = moo.population + moo.archive
        moo.environmental_selection()
        archive = moo.archive
        assert archive == [a, b, d]
        moo.archive = []
        c.fitness = 0.8
        d.fitness = 0.7
        moo.population = [a, b, c, d, e, f]
        moo.joint = moo.population + moo.archive
        moo.environmental_selection()
        archive = moo.archive
        assert archive == [a, c, d]


def main():
    Tests.test_distance_less_than()
    Tests.test_fitness_assignment()
    Tests.test_environmental_selection()

main()

