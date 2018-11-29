"""old deprecated tests that are being kept mostly because they come quite handy for testing new algorithims."""


class TestsNSGA_II:

    # ------------------------tests for PopIndividual-------------------------
    @staticmethod
    def test_dominates():
        # test very simple objective function
        F = [lambda x: x[0]]
        bounds = (-2, 2)
        MOO = NSGA_II(F, bounds, 20)
        for i in range(5):
            MOO.population.append(PopIndividual([i]))
        assert MOO.population[0].dominates(MOO.population[1], MOO.objectives) is True
        assert MOO.population[0].dominates(MOO.population[3], MOO.objectives) is True
        assert MOO.population[3].dominates(MOO.population[0], MOO.objectives) is False
        assert MOO.population[0].dominates(MOO.population[0], MOO.objectives) is False

        # test more complex objective function
        # F = [lambda x: ]
        print("PopIndividaul.dominates() passed")

    @staticmethod
    def test_compare():
        x = PopIndividual([])
        y = PopIndividual([])
        z = PopIndividual([])
        q = PopIndividual([])
        r = PopIndividual([])
        t = PopIndividual([])
        # values don't actually matter, we can just assign empty values

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

        assert x.compare(y) is True
        assert x.compare(z) is False
        assert x.compare(x) is False
        assert q.compare(r) is True
        assert q.compare(t) is False
        print("PopIndividual.compare() passed")

    @staticmethod
    def test_recombine():
        np.random.seed(1002402)
        bounds = [(-20, 20)]
        dist_param = 3
        # test 1
        p1 = PopIndividual([5])
        p2 = PopIndividual([10])
        p1.recombine(p2, dist_param)
        assert np.isclose(p1.values[0], 5.022211826787321, atol=1e-6)
        assert np.isclose(p2.values[0], 9.977788173212678, atol=1e-6)
        # test 2
        p1 = PopIndividual([5])
        p2 = PopIndividual([10])
        p1.recombine(p2, dist_param)
        assert np.isclose(p1.values[0], 5.0, atol=1e-6)
        assert np.isclose(p2.values[0], 10.0, atol=1e-6)
        # test 3
        p1 = PopIndividual([5])
        p2 = PopIndividual([10])
        p1.recombine(p2, dist_param)
        assert np.isclose(p1.values[0], 5.0, atol=1e-6)
        assert np.isclose(p2.values[0], 10.0, atol=1e-6)
        # test 4
        p1 = PopIndividual([5])
        p2 = PopIndividual([10])
        p1.recombine(p2, dist_param)
        assert np.isclose(p1.values[0], 5.998068382956265, atol=1e-6)
        assert np.isclose(p2.values[0], 9.001931617043736, atol=1e-6)
        # test 5
        p1 = PopIndividual([5])
        p2 = PopIndividual([10])
        p1.recombine(p2, dist_param)
        assert np.isclose(p1.values[0], 5.746398124977722, atol=1e-6)
        assert np.isclose(p2.values[0], 9.25360187502228, atol=1e-6)
        print("PopIndividual.recombine() passed")

    @staticmethod
    def test_mutate_individual():
        np.random.seed(12645678)
        bounds = [(-5, 5)]
        distribution_param = 2
        indiv_value = 0
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], -0.07299526594720773, atol=1e-6)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], -0.027816685123562834)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], -0.9019673485855295)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], 0.49704076190606683)
        # test 2
        np.random.seed(12645678)
        indiv_value = 4.9
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], 4.755469373424529)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], 4.844922963455346)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], 3.1141046498006517)
        p = PopIndividual([indiv_value])
        p.mutate_individual(0, bounds, distribution_param)
        assert np.isclose(p.values[0], 4.909940815238122)
        print("PopIndividual.mutate_individual() passed")

    # ----------------------------tests for NSGA-II---------------------------
    @staticmethod
    def test_non_dominated_sort():
        # test very simple objective function
        expected_front = [[[0]], [[1]], [[2]], [[3]], [[4]]]

        F = [lambda x: x[0]]
        bounds = (-2, 2)
        MOO = NSGA_II(F, bounds, parent_pop_size=5)
        for i in range(5):
            MOO.population.append(PopIndividual([i]))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            assert np.all([front[i][0].values == expected_front[i][0]])

        # insert other way to check
        MOO = NSGA_II(F, bounds, parent_pop_size=5)
        for i in range(4, -1, -1):
            MOO.population.append((PopIndividual([i])))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            assert np.all([front[i][0].values == expected_front[i][0]])
        # simple tests done

        # test more complex objective function
        expected_front = [[[1, 1], [2, 1], [3, 1], [4, 1]], [[1, 2], [2, 2], [3, 2], [4, 2]],
                          [[1, 3], [2, 3], [3, 3], [4, 3]], [[1, 4], [2, 4], [3, 4], [4, 4]]]

        F = [lambda x: x[0], lambda x: 1 / x[0] + x[1]]
        MOO = NSGA_II(F, bounds, parent_pop_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                MOO.population.append(PopIndividual([i, j]))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            for j in range(len(expected_front[i])):
                assert np.all([front[i][j].values == expected_front[i][j]])

        # insert other way to check
        MOO = NSGA_II(F, bounds, parent_pop_size=16)
        for i in range(4, 0, -1):
            for j in range(4, 0, -1):
                MOO.population.append(PopIndividual([j, i]))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            sorted_front = sorted(front[i], key=lambda x: x.values[0])
            for j in range(len(expected_front[i])):
                assert np.all([sorted_front[j].values == expected_front[i][j]])
        print("NSGA-II.non_dominated_sort() passed")
        # complex tests passed

    @staticmethod
    def test_crowding_distance_assignment():
        F = [lambda x: x[0], lambda x: 1 / x[0] + x[1]]
        bounds = []
        MOO = NSGA_II(F, bounds, parent_pop_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                MOO.population.append(PopIndividual([i, j]))
        MOO.non_dominated_sort()
        MOO.crowding_distance_assignment(0)
        front1 = MOO.fronts[0]
        d = front1[0]
        c = front1[1]
        b = front1[2]
        a = front1[3]
        assert d.crowding_distance == np.inf
        assert a.crowding_distance == np.inf
        assert math.isclose(c.crowding_distance, 1)
        assert math.isclose(b.crowding_distance, 14 / 9)
        print("NSGA-II.crowding_distance_assignment() passed")

    @staticmethod
    def test_crowded_sort():
        """unfinished"""
        F = lambda x: x[0]
        bounds = (-10, 10)
        MOO = NSGA_II(F, bounds)
        a = PopIndividual([1])
        b = PopIndividual([2])
        c = PopIndividual([3])
        d = PopIndividual([4])
        e = PopIndividual([5])
        f = PopIndividual([6])
        g = PopIndividual([7])
        h = PopIndividual([8])
        a.rank = 1
        a.crowding_distance = 10
        b.rank = 1
        b.crowding_distance = 9
        c.rank = 1
        c.crowding_distance = 7
        d.rank = 2
        d.crowding_distance = 15
        e.rank = 2
        e.crowding_distance = 14
        f.rank = 2
        f.crowding_distance = 10
        g.rank = 3
        g.crowding_distance = 25
        h.rank = 7
        h.crowding_distance = 10
        MOO.fronts.append([a, b, c, d, e, f, g, h])
        # setup complete
        expected = [a, b, c, d, e, f, g, h]
        for i in range(10):
            np.random.shuffle(MOO.fronts[0])
            MOO.crowded_sort(0)
            assert MOO.fronts[0] == expected
        print("NSGA-II.crowded_sort() passed")

    @staticmethod
    def test_tournament_selection():
        np.random.seed(12645678)
        F = lambda x: x[0]
        bounds = (-10, 10)
        MOO = NSGA_II(F, bounds, parent_pop_size=4)
        a = PopIndividual([1])
        b = PopIndividual([2])
        c = PopIndividual([3])
        d = PopIndividual([4])
        e = PopIndividual([5])
        a.rank = 1
        a.crowding_distance = 3
        b.rank = 1
        b.crowding_distance = 2
        c.rank = 4
        c.crowding_distance = 7
        d.rank = 5
        d.crowding_distance = 1
        e.rank = 4
        e.crowding_distance = 2
        MOO.parent_pop = [a, b, c, d]
        MOO.tournament_selection()
        assert MOO.child_pop == [c, a, a, c]
        np.random.seed(12345678)
        MOO = NSGA_II(F, bounds, parent_pop_size=5)
        MOO.parent_pop = [a, b, c, d, e]
        MOO.tournament_selection()
        assert MOO.child_pop == [b, a, a, b, e]
        print("NSGA-II.tournament_sort() passed")

    @staticmethod
    def test_crossover():
        np.random.seed(1234567890)
        a = PopIndividual([5, 3])
        b = PopIndividual([2, 7])
        c = PopIndividual([20, 91])
        d = PopIndividual([-3, -17])
        e = PopIndividual([7, 8])
        F = lambda x: x[0]
        bounds = [(-5, 20), (-20, 100)]
        MOO = NSGA_II(F, bounds)
        MOO.child_pop = [a, b, c, d, e]
        MOO.crossover()
        assert np.all(np.isclose(MOO.child_pop[0].values, PopIndividual([5, 3]).values))
        assert np.all(np.isclose(MOO.child_pop[1].values, PopIndividual([2, 7.0]).values))
        assert np.all(np.isclose(MOO.child_pop[2].values, PopIndividual([20.268629037161734, 91]).values))
        assert np.all(np.isclose(MOO.child_pop[3].values, PopIndividual([-2.3824405709323537, -17]).values))
        assert np.all(np.isclose(MOO.child_pop[4].values, PopIndividual([6.113811533770617, 8]).values))
        print("NSGA-II.crossover() passed")

    @staticmethod
    def test_mutate():
        np.random.seed(123456789)
        a = PopIndividual([5, 3])
        b = PopIndividual([2, 7])
        c = PopIndividual([20, 91])
        d = PopIndividual([-3, -17])
        e = PopIndividual([7, 8])
        F = lambda x: x[0]
        bounds = [(-100, 100), (-100, 100)]
        MOO = NSGA_II(F, bounds, mut_prob=0.5, parent_pop_size=5)
        MOO.child_pop = [a, b, c, d, e]
        MOO.mutate()
        assert np.all(np.isclose(MOO.child_pop[0].values, PopIndividual([5.207918232261214, 3]).values))
        assert np.all(np.isclose(MOO.child_pop[1].values, PopIndividual([2.0609653247733437, 7]).values))
        assert np.all(np.isclose(MOO.child_pop[2].values, PopIndividual([20, 86.94314294871329]).values))
        assert np.all(np.isclose(MOO.child_pop[3].values, PopIndividual([-3, -17]).values))
        assert np.all(np.isclose(MOO.child_pop[4].values, PopIndividual([11.278392367467381, 9.229767723357321]).values))
        print("NSGA-II.mutate() passed")


class TestsSPEA:

    @staticmethod
    def test_cluster_distance():
        objectives = (lambda x: x[0], lambda x: x[1])
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds)
        a = PopIndividual([1, 1], objectives)
        b = PopIndividual([1, 2], objectives)
        c = PopIndividual([2, 1], objectives)
        d = PopIndividual([2, 2], objectives)
        distance_dict = {(a, c): 1, (a, d): np.sqrt(2), (b, c): np.sqrt(2), (b, d): 1}
        expected = (np.sqrt(2) + 1) / 2
        assert np.isclose(moo.cluster_distance([a, b], [c, d], distance_dict), expected)

    @staticmethod
    def test_cluster_centroid():
        objectives = (lambda x: x[0], lambda x: x[1])
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds)
        a = PopIndividual([-1, 1], objectives)
        b = PopIndividual([1, 1], objectives)
        c = PopIndividual([0, -np.sqrt(2)], objectives)
        d = PopIndividual([0, 0], objectives)
        distance_dict = {(a, b): 2, (a, c): np.sqrt(1 + (1 + np.sqrt(2)) ** 2), (a, d): np.sqrt(2),
                         (b, c): np.sqrt(1 + (1 + np.sqrt(2)) ** 2), (b, d): np.sqrt(2), (c, d): np.sqrt(2)}
        assert moo.cluster_centroid([a, b, c, d], distance_dict) == d

    @staticmethod
    def test_reduce_archive():
        objectives = [lambda x: x[0], lambda x: 1/x[0] + x[1]]
        bounds = [(1, 4), (1, 4)]
        moo = SPEA(objectives, bounds, archive_size=2)
        population = []
        for i in range(1, 4):
            population.append(PopIndividual([1, i], objectives))
            population.append(PopIndividual([1.5, i], objectives))
            population.append(PopIndividual([3.5, i], objectives))
            population.append(PopIndividual([4, i], objectives))
        population.append(PopIndividual([1.25, 2], objectives))
        population.append(PopIndividual([3.75, 2], objectives))
        moo.archive = population
        moo.reduce_archive()
        assert population[-1] in moo.archive
        assert population[-2] in moo.archive
        assert len(moo.archive) == 2

    @staticmethod
    def test_fitness_assignment():
        objectives = [lambda x: x[0], lambda x: x[1]]
        bounds = [(1, 4), (1, 4)]
        population = []
        archive = []
        archive.append(PopIndividual([1, 5], objectives))
        archive.append(PopIndividual([2, 4], objectives))
        archive.append(PopIndividual([3, 2], objectives))
        population.append(PopIndividual([1.5, 7], objectives))
        population.append(PopIndividual([1.2, 6], objectives))
        population.append(PopIndividual([1.9, 5.4], objectives))
        population.append(PopIndividual([2.2, 4.9], objectives))
        population.append(PopIndividual([2.7, 4.1], objectives))
        population.append(PopIndividual([3.2, 2.9], objectives))
        population.append(PopIndividual([3.6, 4.4], objectives))
        population.append(PopIndividual([3.1, 5.7], objectives))
        moo = SPEA(objectives, bounds, population_size=len(population), archive_size=len(archive))
        moo.archive = archive
        moo.population = population
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
    def test_fast_non_dominated():
        expected_front = [[0], [1], [2], [3], [4]]

        objectives = [lambda x: x[0]]
        bounds = (-2, 2)
        moo = SPEA(objectives, bounds, population_size=5)
        for i in range(5):
            moo.population.append(PopIndividual([i], objectives))
        front = moo.fast_non_dominated(moo.population)
        for i in range(len(front)):
            assert np.all(front[i].values == expected_front[i])
        # test the other way round
        moo = SPEA(objectives, bounds, population_size=5)
        for i in range(4, -1, -1):
            moo.population.append(PopIndividual([i], objectives))
        front = moo.fast_non_dominated(moo.population)
        for i in range(len(front)):
            assert np.all(front[i].values == expected_front[i])

        expected_front = [[1, 1], [2, 1], [3, 1], [4, 1]]

        objectives = [lambda x: x[0], lambda x: 1 / x[0] + x[1]]
        moo = SPEA(objectives, bounds, population_size=16)
        for i in range(1, 5):
            for j in range(1, 5):
                moo.population.append(PopIndividual([i, j], objectives))
        front = moo.fast_non_dominated(moo.population)
        front.sort(key=lambda x: x.values[0])
        for i in range(len(front)):
            assert np.all(np.isclose(front[i].values, expected_front[i]))


class TestsAbstractMOEA:

    @staticmethod
    def test_tournament_selection():
        np.random.seed(12345678)
        objectives = [lambda x: x[0]]
        bounds = (-10, 10)
        moo = AbstractMOEA(objectives, bounds)
        a = AbstractPopIndividual([1], objectives)
        b = AbstractPopIndividual([2], objectives)
        c = AbstractPopIndividual([3], objectives)
        d = AbstractPopIndividual([4], objectives)
        e = AbstractPopIndividual([5], objectives)
        a.fitness = 1
        b.fitness = 2
        c.fitness = 3
        d.fitness = 4
        e.fitness = 5
        expected = [b, a, a, b, e]
        child = moo.tournament_selection([a, b, c, d, e], 5)
        i = 0
        while i < len(child):
            assert np.isclose(child[i].values, expected[i].values)
            i += 1
        print("NSGA-II.tournament_sort() passed")
