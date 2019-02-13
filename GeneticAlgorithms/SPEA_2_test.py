
"""
Tests for SPEA_2
"""

from GeneticAlgorithms.SPEA_2 import *
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class PopIndividualTests:

    @staticmethod
    def test_distance_less_than():
        def objectives(x):
            return np.array([x[0]])
        a = PopIndividual([1])
        b = PopIndividual([2])
        c = PopIndividual([3])
        d = PopIndividual([4])
        moo = AbstractMOEA(objectives, None, 1)
        population = [a, b, c, d]
        moo.run_model(population)
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
        def objectives(x):
            return np.array([x[0]])
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


class SPEA_2Tests:

    @staticmethod
    def test_fitness_assignment():
        def objectives(x):
            return np.array([x[0], 1/x[0] + x[1]])
        bounds = [(-1, 1), (-1, 1)]
        moo = SPEA_2(objectives, bounds, population_size=9, archive_size=0, number_objectives=2)
        for i in range(1, 4):
            for j in range(1, 4):
                moo.population.append(PopIndividual([j, i]))
        moo.joint = moo.population + moo.archive
        moo.run_model(moo.joint)
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
        def objectives(x):
            return np.array([x[0]])
        bounds = []
        moo = SPEA_2(objectives, bounds, 1)
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
        moo.next_archive = [a, b, c, d]
        moo.distance_sort()
        moo.truncate()
        assert moo.next_archive == [a, b, d]
        moo.distance_sort()
        moo.truncate()
        assert moo.next_archive == [b, d]
        moo.distance_sort()
        moo.truncate()
        assert moo.next_archive == [d]

    @staticmethod
    def test_distance_assignment():
        def objectives(x):
            return np.array([x[0]])
        bounds = []
        moo = SPEA_2(objectives, bounds, 1)
        a = PopIndividual([1])
        b = PopIndividual([2])
        c = PopIndividual([3])
        d = PopIndividual([4])
        moo.next_archive = [a, b, c, d]
        moo.run_model(moo.next_archive)
        moo.distance_assignment()
        assert moo.next_archive[0].distances[b] == 1
        assert moo.next_archive[0].distances[c] == 2
        assert moo.next_archive[0].distances[d] == 3
        assert moo.next_archive[1].distances[a] == 1
        assert moo.next_archive[1].distances[c] == 1
        assert moo.next_archive[1].distances[d] == 2
        assert moo.next_archive[2].distances[a] == 2
        assert moo.next_archive[2].distances[b] == 1
        assert moo.next_archive[2].distances[d] == 1
        assert moo.next_archive[3].distances[a] == 3
        assert moo.next_archive[3].distances[c] == 1
        assert moo.next_archive[3].distances[b] == 2

    @staticmethod
    def test_environmental_selection():
        def objectives(x):
            return np.array([x[0], 1/x[0] + x[1]])
        bounds = []
        moo = SPEA_2(objectives, bounds, 2, population_size=4, archive_size=4)
        individuals = []
        for i in range(1, 7):
            individuals.append(PopIndividual([i, 1]))
            individuals[i - 1].fitness = 0.1
        moo.joint = individuals.copy()
        moo.run_model(moo.joint)
        moo.environmental_selection()
        assert moo.archive == individuals[:2] + [individuals[3]] + [individuals[5]]


if __name__ == '__main__':
    tests = [PopIndividualTests.test_distance_less_than, PopIndividualTests.test_graph_operations,
             SPEA_2Tests.test_fitness_assignment, SPEA_2Tests.test_truncate, SPEA_2Tests.test_distance_assignment,
             SPEA_2Tests.test_environmental_selection]
    for test in tests:
        test()

