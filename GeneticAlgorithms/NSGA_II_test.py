
"""
Tests NSGA_II
"""

from GeneticAlgorithms.NSGA_II import *


class PopIndividualTests:

    @staticmethod
    def test__lt__():
        x = PopIndividual([1])
        y = PopIndividual([2])
        z = PopIndividual([3])
        q = PopIndividual([4])
        r = PopIndividual([5])
        t = PopIndividual([6])
        x.rank = 1
        x.crowding_distance = 5
        y.rank = 1
        y.crowding_distance = 4
        z.rank = 1
        z.crowding_distance = 6
        q.rank = 2
        q.crowding_distance = 2
        r.rank = 3
        r.crowding_distance = 1
        t.rank = 1
        t.crowding_distance = 3
        assert (x < y) is True
        assert (x < z) is False
        assert (x < x) is False
        assert (q < r) is True
        assert (q < t) is False
        print('__lt__() passed')


class NSGA_IITests:

    @staticmethod
    def test_non_dominated_sort():
        # test very simple objective function
        def objectives(x):
            return np.array(x[0])
        expected_front = [[[0]], [[1]], [[2]], [[3]], [[4]]]
        bounds = [(-2, 2)]
        moo = NSGA_II(objectives, bounds, 1, parent_pop_size=5)
        for i in range(5):
            moo.population.append(PopIndividual([i]))
        moo.run_model(moo.population)
        moo.non_dominated_sort()
        front = moo.fronts
        for i in range(len(expected_front)):
            assert np.all([front[i][0].d_vars == expected_front[i][0]])
        # insert other way to check
        moo = NSGA_II(objectives, bounds, 1, parent_pop_size=5)
        for i in range(4, -1, -1):
            moo.population.append((PopIndividual([i])))
        moo.run_model(moo.population)
        moo.non_dominated_sort()
        front = moo.fronts
        for i in range(len(expected_front)):
            assert np.all([front[i][0].d_vars == expected_front[i][0]])
        expected_front = [[(1, 1), (2, 1), (3, 1), (4, 1)], [(1, 2), (2, 2), (3, 2), (4, 2)],
                          [(1, 3), (2, 3), (3, 3), (4, 3)], [(1, 4), (2, 4), (3, 4), (4, 4)]]
        # -----------------------------------------------------------------------------------
        # change to more complex objective function
        # -----------------------------------------------------------------------------------
        def objectives(x):
            return np.array([x[0], 1 / x[0] + x[1]])
        moo = NSGA_II(objectives, bounds, 2, parent_pop_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                moo.population.append(PopIndividual([i, j]))
        moo.run_model(moo.population)
        moo.non_dominated_sort()
        front = moo.fronts
        for i in range(len(expected_front)):
            set_version = set()
            for j in range(len(expected_front[i])):
                set_version.add(tuple(front[i][j].d_vars))
            assert set_version == set(expected_front[i])
        moo = NSGA_II(objectives, bounds, 2, parent_pop_size=16)
        for i in range(4, 0, -1):
            for j in range(4, 0, -1):
                moo.population.append(PopIndividual([j, i]))
        moo.run_model(moo.population)
        moo.non_dominated_sort()
        front = moo.fronts
        for i in range(len(expected_front)):
            set_version = set()
            for j in range(len(expected_front[i])):
                set_version.add(tuple(front[i][j].d_vars))
            assert set_version == set(expected_front[i])
        print('non_dominated_sort() passed')

    @staticmethod
    def test_crowding_distance_assignment():
        def objectives(x):
            return np.array([x[0], 1 / x[0] + x[1]])
        bounds = []
        moo = NSGA_II(objectives, bounds, 2, parent_pop_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                moo.population.append(PopIndividual([i, j]))
        moo.run_model(moo.population)
        moo.non_dominated_sort()
        fronts = moo.fronts
        moo.crowding_distance_assignment(fronts[0])
        front1 = moo.fronts[0]
        d = front1[0]
        c = front1[1]
        b = front1[2]
        a = front1[3]
        assert d.crowding_distance == np.inf
        assert a.crowding_distance == np.inf
        assert np.isclose(c.crowding_distance, 1)
        assert np.isclose(b.crowding_distance, 14 / 9)
        print("crowding_distance_assignment passed")

    @staticmethod
    def test_run():
        def objectives(x):
            return np.array([x[0], x[1]])
        bounds = [(0, 1), (0, 1)]
        moo_current = NSGA_II(objectives=objectives, bounds=bounds, number_objectives=2, parent_pop_size=10,
                              iterations=3)
        model = 'test1'
        model_call = ['python', os.path.join(os.getcwd(), 'TestProblems', 'StochasticProblemSuite.py')]
        def objective_error():
            raise Exception("objective should not be called")
        moo_new = NSGA_II(bounds=bounds, number_objectives=2, model=model, pst=model_call, objectives=objective_error,
                          parent_pop_size=10, iterations=3)
        np.random.seed(12323)
        front = moo_current.run()
        print(Population(front, constrained=False))
        np.random.seed(12323)
        print()
        front = moo_new.run_IO()
        print(front)
        print('run passed')


class PopulationTests:

    @staticmethod
    def test_non_dominated_sort():
        # test very simple objective function
        def objectives(x):
            return np.array(x[0])
        expected_front = [[[0]], [[1]], [[2]], [[3]], [[4]]]
        bounds = [(-2, 2)]
        pst = ['python', os.path.join(os.getcwd(), 'TestProblems', 'StochasticProblemSuite.py')]
        model = 'test2'
        moo = NSGA_II(objectives=None, bounds=bounds, number_objectives=1 ,model=model, pst=pst, parent_pop_size=5)
        population = Population([PopIndividual([i]) for i in range(5)], constrained=False)
        moo.run_model_IO(population=population)
        fronts = population.non_dominated_sort()
        for i in range(len(expected_front)):
            assert np.all([fronts[i][0].d_vars == expected_front[i][0]])
        # insert other way to check
        population = Population([PopIndividual([i]) for i in range(4, -1, -1)], constrained=False)
        moo.run_model_IO(population=population)
        fronts = population.non_dominated_sort()
        for i in range(len(expected_front)):
            assert np.all([fronts[i][0].d_vars == expected_front[i][0]])
            # -----------------------------------------------------------------------------------
            # change to more complex objective function
            # -----------------------------------------------------------------------------------
        expected_front = [[(1, 1), (2, 1), (3, 1), (4, 1)], [(1, 2), (2, 2), (3, 2), (4, 2)],
                          [(1, 3), (2, 3), (3, 3), (4, 3)], [(1, 4), (2, 4), (3, 4), (4, 4)]]
        model = 'simple'
        moo = NSGA_II(objectives=None, bounds=bounds, number_objectives=2, model=model, pst=pst, parent_pop_size=16)
        population = []
        for i in range(1, 5):
            for j in range(1, 5):
                population.append(PopIndividual([i, j]))
        population = Population(population, constrained=False)
        moo.run_model_IO(population=population)
        fronts = population.non_dominated_sort()
        for i in range(len(expected_front)):
            set_version = set()
            for j in range(len(expected_front[i])):
                set_version.add(tuple(fronts[i][j].d_vars))
            assert set_version == set(expected_front[i])
        print('non_dominated_sort passed')


if __name__ == '__main__':
    tests = [PopIndividualTests.test__lt__, NSGA_IITests.test_non_dominated_sort,
             NSGA_IITests.test_crowding_distance_assignment, PopulationTests.test_non_dominated_sort,
             NSGA_IITests.test_run]
    for test in tests:
        test()