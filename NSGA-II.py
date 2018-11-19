# NSGA-II algorithim for multi objective optimisation
# benchmark testing for SAM MOOUU using Differential evolution
# version proposed by Deb[2002] IEEE
# GNS.cri
import numpy as np
import math


# import byteManipulation as byt

class NSGA_II:
    """NSGA-II algorithm"""

    # ------------------------External methods--------------------------------
    def __init__(self, objectives, bounds, pop_size=50, cross_prob=0.5, cross_dist=20, mut_prob=0.05, mut_dist=30):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           pop_size: population size
           cross_prob: probability of crossover occurring for any child
           mut_prob: probability of mutation occuring for any child"""
        self.cross_dist = cross_dist
        self.mut_dist = mut_dist
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.objectives = objectives
        self.bounds = bounds
        self.population = []
        self.fronts = []
        self.pareto_optimal_set = []
        self.parent_pop = []
        self.child_pop = []
        self.pop_size = pop_size
        self.n_decision_variables = len(bounds)

    def main_algorithm(self):
        self.population = []
        self.population.append(self.parent_pop)
        self.population.append(self.child_pop)
        self.non_dominated_sort()
        self.parent_pop = []
        front_index = 0
        while len(self.parent_pop) + len(self.fronts[front_index]) <= self.pop_size:
            self.crowding_distance_assignment(front_index)
            self.parent_pop.append(self.fronts[front_index])
            front_index += 1

    # ------------------------internal methods-------------------------------
    def non_dominated_sort(self):
        """ ----------------------------------Internal Method-----------------------------------
            Sorts the population into a set of non-dominated pareto fronts F1, F2 ...
        """
        self.fronts.append([])
        for i in range(0, len(self.population)):
            p = self.population[i]
            for j in range(0, len(self.population)):
                q = self.population[j]
                if p.dominates(q, self.objectives):
                    p.dominated_set.append(q)
                elif q.dominates(p, self.objectives):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                self.fronts[0].append(p)
        i = 0
        while self.fronts[i] != []:
            Q = []
            for p in self.fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            self.fronts.append(Q)
        self.fronts.pop()

    def crowding_distance_assignment(self, front_index):
        """Calculates the crowding distance for each individual in the population"""
        front = self.fronts[front_index]
        n = len(front)
        for f in self.objectives:
            sort_key = lambda individual: f(individual.values)
            front.sort(key=sort_key)
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            fmax = f(front[-1].values)
            fmin = f(front[0].values)
            for i in range(1, n - 1):
                front[i].crowding_distance = front[i].crowding_distance + (
                            f(front[i + 1].values) - f(front[i - 1].values)) \
                                             / (fmax - fmin)

    def crowded_sort(self, front_index):
        """implements an insertion sort using the crowded comparison operator"""
        for i in range(1, len(self.fronts[front_index])):
            indiv_1 = self.fronts[front_index][i]
            j = i - 1
            in_place = False
            while j > 0 and not in_place:
                if indiv_1.compare(self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                else:
                    self.fronts[front_index][j + 1] = indiv_1
                    in_place = True
                j -= 1
            if j == 0 and not in_place:
                if indiv_1.compare(self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                    self.fronts[front_index][j] = indiv_1
                else:
                    self.fronts[front_index][j + 1] = indiv_1

    def tournament_selection(self):
        """performs a tournament selection on the parent population. Note that it is assumed that the population has
        already been ranked and been assign"""
        np.random.shuffle(self.parent_pop)
        i = 0
        N = len(self.parent_pop)
        while i < (N // 2):
            if self.parent_pop[2 * i].compare(self.parent_pop[2 * i + 1]):
                self.child_pop.append(self.parent_pop[2 * i])
            else:
                self.child_pop.append(self.parent_pop[2 * i + 1])
            i += 1
        # now repeat this process again:
        i = 0
        np.random.shuffle(self.parent_pop)
        while i < (N // 2):
            if self.parent_pop[2 * i].compare(self.parent_pop[2 * i + 1]):
                self.child_pop.append(self.parent_pop[2 * i])
            else:
                self.child_pop.append(self.parent_pop[2 * i + 1])
            i += 1
        if N % 2 == 1:  # i.e is odd
            self.child_pop.append(self.parent_pop[N - 1])
        # child population now populated from tournament selection

    def crossover(self):
        for i in range(len(self.child_pop)):
            if np.random.random() > self.cross_prob:
                rand_index = np.random.randint(0, len(self.child_pop))
                self.child_pop[i].recombine(self.child_pop[rand_index], self.cross_dist)

    def mutate(self):
        """use mutation clock and polynomial mutation"""
        k = 0
        i = 0
        self.child_pop[i].mutate_individual(k, self.bounds, self.mut_dist)
        while i < self.pop_size:
            variable_step = - 1 / self.mut_prob * np.log(1 - np.random.random())
            i = i + np.ceil((k + variable_step) / self.n_decision_variables)
            k = (k + variable_step) % self.n_decision_variables
            self.child_pop[i].mutate_individual(k, self.bounds, self.mut_dist)


# --------------------------Other Classes--------------------------------------
class PopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)
        # self.byte_encoding = self._encode(value)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def dominates(self, other, objectives):
        i = 0
        weak_dom = True
        strong_condition = False
        while i < len(objectives):
            if objectives[i](self.values) > objectives[i](other.values):
                weak_dom = False
                i = len(objectives)
            elif objectives[i](self.values) < objectives[i](other.values):
                strong_condition = True
            i += 1
        return weak_dom and strong_condition

    def __str__(self):
        s = "solution at {}, rank {}".format(self.values, self.rank)
        return s

    def __repr__(self):
        return str(self.values)

    # note: some methods for encoding and decoding floats are included here. These may be used at a later date
    # for now it has been determined that for large sets of decision variables, simply swapping these may be sufficient.
    # def _encode(self, values):
    #     encoded = []
    #     for value in values:
    #         encoded.append(byt.float2bytearray(value))
    #     return encoded

    # def _decode(self, values):
    #     decoded = []
    #     for value in values:
    #         decoded.append(byt.bytearray2float(value))
    #     return decoded

    def compare(self, other):
        """true if self is better, false otherwise"""
        return ((self.rank < other.rank) or (
                    (self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def recombine(self, other, distribution_parameter):
        """Use SBX Deb [1995] to cross self and other. Note this uses the suggested extension to multi variable problems
        given by Deb and Kumar [1995], where variables in each of the individuals crossing have P = 0.5 of crossing"""
        for i in range(len(self.values)):
            if np.random.random() >= 0.5:
                r = np.random.random()
                if r <= 0.5:
                    beta = (2 * r) ** (1 / (distribution_parameter + 1))
                else:
                    beta = (1 / (2 - 2 * r)) ** (1 / (distribution_parameter + 1))
                p1 = self.values[i]
                p2 = other.values[i]
                self.values[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                other.values[i] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    def mutate_individual(self, variable, bounds, distribution_parameter):
        """use polynomial mutation to mutate self"""
        u = np.random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.values[variable] = self.values[variable] + delta * (self.values[variable] - bounds[variable][0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.values[variable] = self.values[variable] + delta * (bounds[variable][1] - self.values[variable])


# ----------------------------Tests-----------------------
class Tests:

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
        MOO = NSGA_II(F, bounds)
        for i in range(5):
            MOO.population.append(PopIndividual([i]))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            assert np.all([front[i][0].values == expected_front[i][0]])

        # insert other way to check
        MOO = NSGA_II(F, bounds)
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
        MOO = NSGA_II(F, bounds)
        for i in range(1, 5):
            for j in range(1, 5):
                MOO.population.append(PopIndividual([i, j]))
        MOO.non_dominated_sort()
        front = MOO.fronts
        for i in range(len(expected_front)):
            for j in range(len(expected_front[i])):
                assert np.all([front[i][j].values == expected_front[i][j]])

        # insert other way to check
        MOO = NSGA_II(F, bounds)
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
        MOO = NSGA_II(F, bounds, 20)
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
        MOO = NSGA_II(F, bounds)
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
        MOO = NSGA_II(F, bounds)
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
        pass


def main():
    # PopIndividual tests
    Tests.test_dominates()
    Tests.test_compare()
    Tests.test_recombine()
    Tests.test_mutate_individual()

    # NSGA-II tests
    Tests.test_non_dominated_sort()
    Tests.test_crowding_distance_assignment()
    Tests.test_crowded_sort()
    Tests.test_tournament_selection()
    Tests.test_crossover()


main()
