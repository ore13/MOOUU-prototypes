
"""
Tests for SPEA
"""

from GeneticAlgorithms.SPEA import *
import argparse


class SPEATests:

    @staticmethod
    def test_cluster_distance():
        def objectives(x):
            return np.array([x[0], x[1]])
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds, 2)
        a = PopIndividual([1, 1])
        b = PopIndividual([1, 2])
        c = PopIndividual([2, 1])
        d = PopIndividual([2, 2])
        distance_dict = {(a, c): 1, (a, d): np.sqrt(2), (b, c): np.sqrt(2), (b, d): 1}
        expected = (np.sqrt(2) + 1) / 2
        assert np.isclose(moo.cluster_distance([a, b], [c, d], distance_dict), expected)

    @staticmethod
    def test_cluster_centroid():
        def objectives(x):
            return np.array([x[0], x[1]])
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds, 2)
        a = PopIndividual([-1, 1])
        b = PopIndividual([1, 1])
        c = PopIndividual([0, -np.sqrt(2)])
        d = PopIndividual([0, 0])
        moo.run_model([a, b, c, d])
        distance_dict = {(a, b): 2, (a, c): np.sqrt(1 + (1 + np.sqrt(2)) ** 2), (a, d): np.sqrt(2),
                         (b, c): np.sqrt(1 + (1 + np.sqrt(2)) ** 2), (b, d): np.sqrt(2), (c, d): np.sqrt(2)}
        assert moo.cluster_centroid([a, b, c, d], distance_dict) == d

    @staticmethod
    def test_reduce_archive():
        def objectives(x):
            return np.array([x[0], x[1]])
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds, 2, archive_size=2)
        population = []
        for i in range(1, 4):
            population.append(PopIndividual([1, i]))
            population.append(PopIndividual([1.5, i]))
            population.append(PopIndividual([3.5, i]))
            population.append(PopIndividual([4, i]))
        population.append(PopIndividual([1.25, 2]))
        population.append(PopIndividual([3.75, 2]))
        moo.run_model(population)
        moo.archive = population
        moo.reduce_archive()
        assert population[-1] in moo.archive
        assert population[-2] in moo.archive
        assert len(moo.archive) == 2

    @staticmethod
    def test_fitness_assignment():
        def objectives(x):
            return np.array([x[0], x[1]])
        bounds = [(1, 4), (1, 4)]
        population = []
        archive = []
        archive.append(PopIndividual([1, 5]))
        archive.append(PopIndividual([2, 4]))
        archive.append(PopIndividual([3, 2]))
        population.append(PopIndividual([1.5, 7]))
        population.append(PopIndividual([1.2, 6]))
        population.append(PopIndividual([1.9, 5.4]))
        population.append(PopIndividual([2.2, 4.9]))
        population.append(PopIndividual([2.7, 4.1]))
        population.append(PopIndividual([3.2, 2.9]))
        population.append(PopIndividual([3.6, 4.4]))
        population.append(PopIndividual([3.1, 5.7]))
        moo = SPEA(objectives, bounds, 2, population_size=len(population), archive_size=len(archive))
        moo.archive = archive
        moo.population = population
        moo.run_model(population)
        moo.run_model(archive)
        moo.fitness_assignment()
        assert np.isclose(moo.archive[0].fitness, 4/9)
        assert np.isclose(moo.archive[1].fitness, 4/9)
        assert np.isclose(moo.archive[2].fitness, 3/9)
        for i in range(5):
            assert np.isclose(moo.population[i].fitness, 13/9)
        assert np.isclose(moo.population[5].fitness, 12/9)
        assert np.isclose(moo.population[6].fitness, 16/9)
        assert np.isclose(moo.population[7].fitness, 20/9)

    @staticmethod
    def test_fast_non_dominated_front():
        def objectives(x):
            return np.array([x[0]])
        expected_front = [[0], [1], [2], [3], [4]]
        bounds = [(-2, 2)]
        moo = SPEA(objectives, bounds, 1, population_size=5)
        for i in range(5):
            moo.population.append(PopIndividual([i]))
        moo.run_model(moo.population)
        front = moo.fast_non_dominated_front(moo.population)
        for i in range(len(front)):
            assert np.all(front[i].d_vars == expected_front[i])
        moo = SPEA(objectives, bounds, 1, population_size=5)
        for i in range(4, -1, -1):
            moo.population.append(PopIndividual([i], objectives))
        moo.run_model(moo.population)
        front = moo.fast_non_dominated_front(moo.population)
        for i in range(len(front)):
            assert np.all(front[i].d_vars == expected_front[i])
        expected_front = [[1, 1], [2, 1], [3, 1], [4, 1]]
        # ------------------------------------------------------
        # test using a more complex objective function
        # ------------------------------------------------------
        def objectives(x):
            return np.array([x[0], 1/x[0] + x[1]])
        moo = SPEA(objectives, bounds, 2, population_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                moo.population.append(PopIndividual([i, j], objectives))
        moo.run_model(moo.population)
        front = moo.fast_non_dominated_front(moo.population)
        front.sort(key=lambda x: x.d_vars[0])
        for i in range(len(front)):
            assert np.all(np.isclose(front[i].d_vars, expected_front[i]))


if __name__ == '__main__':
    tests = [SPEATests.test_cluster_distance, SPEATests.test_cluster_centroid, SPEATests.test_reduce_archive,
             SPEATests.test_fitness_assignment, SPEATests.test_fast_non_dominated_front]
    for test in tests:
        test()
