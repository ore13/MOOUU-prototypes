"""
NSGA-II algorithim for multi objective optimisation
benchmark testing for SAM MOOUU using Differential evolution
version proposed by Deb[2002] IEEE
GNS.cri
Otis Rea
"""
import numpy as np
from Abstract_Moo import *


class NSGA_II(AbstractMOEA):
    """implements the NSGA_II algorithm for multi objective optimisation
       Methods to be used:
       __init__: initialise the algorithm.
            Objectives: a list of functions to be optimised, e.g [f1, f2...]
            bounds: limits on the decision variables in the problem, to be specified as
            ([x1_lower, x1_upper], [x2_lower, x2_upper], ...)
            Other d_vars are just numbers specifying parameters of the algorithm
        run: runs the algorithm and returns an approximation to the pareto front
    """

    # ------------------------External methods--------------------------------
    def __init__(self, objectives, bounds, number_objectives,
                 constraints=None, parent_pop_size=100, cross_prob=0.9, cross_dist=15,mut_prob=0.01, mut_dist=20,
                 iterations=20):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions to be optimised
           bounds: array of upper and lower bounds for each decision variable, eg [(0, 5), (-2, 2)]
           ------------------------------------- Optional parameters -------------------------------------------------
           parent_pop_size: size of parent population. Full population size is 2 * parent_pop due to child population
           cross_prob: probability of crossover occurring for any child
           cross_dist: distribution parameter of the crossover operation
           mut_prob: probability of mutation occurring for any child
           mut_dist: distribution parameter of the mutation operation
           iterations: number of iterations of the algorithm
        """
        super().__init__(objectives, bounds, number_objectives, constraints, cross_prob, cross_dist, mut_prob,
                         mut_dist, iterations)
        # --------------------------------sets of population members------------------------------------------
        self.parent_pop_size = parent_pop_size
        self.parent_pop = []
        self.child_pop = []
        self.fronts = []
        self.population = []

    def __str__(self):
        return "NSGA-II"

    def run(self):
        """Run the NSGA-II algorithm. This will return an approximation to the pareto front"""
        self.parent_pop = self.initialise_population(self.parent_pop_size, PopIndividual)
        self.non_dominated_sort(init=True)
        self.child_pop = []
        self.child_pop = self.tournament_selection(self.parent_pop, self.parent_pop_size)
        self.crossover_step_SBX(self.child_pop)
        self.mutation_step_polynomial(self.child_pop)
        for i in range(self.iterations):
            self.population = self.child_pop + self.parent_pop
            self.non_dominated_sort()
            self.parent_pop = []
            j = 0
            while len(self.parent_pop) + len(self.fronts[j]) < self.parent_pop_size:
                self.crowding_distance_assignment(j)
                self.parent_pop += self.fronts[j]
                j += 1
            self.crowding_distance_assignment(j)
            self.fronts[j].sort()
            self.parent_pop += self.fronts[j][:(self.parent_pop_size - len(self.parent_pop))]
            self.child_pop = []
            self.child_pop = self.tournament_selection(self.parent_pop, self.parent_pop_size,
                                                       is_better=lambda x, y: x < y)
            self.crossover_step_SBX(self.child_pop)
            self.mutation_step_polynomial(self.child_pop)
            self.reset_population(self.parent_pop)
        return self.fronts[0]

    def non_dominated_sort(self, init=False):
        """Internal method. Sorts the population into a set of non-dominated pareto fronts F1, F2 ..."""
        if init:
            self.population = self.parent_pop
        self.fronts = []
        Q = set()
        for i in range(len(self.population)):
            p = self.population[i]
            for j in range(i + 1, len(self.population)):
                q = self.population[j]
                if self.dominates(p, q):
                    p.dominated_set.append(q)
                    q.domination_count += 1
                elif self.dominates(q, p):
                    q.dominated_set.append(p)
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                Q.add(p)
        self.fronts.append(list(Q))
        i = 0
        while self.fronts[i]:
            Q = set()
            for p in self.fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        Q.add(q)
            i += 1
            self.fronts.append(list(Q))
        self.fronts.pop()

    def crowding_distance_assignment(self, front_index):
        """Internal method. Calculates and assigns the crowding distance for each individual in the population"""
        front = self.fronts[front_index]
        i = 0
        while i < self.number_objectives:
            front.sort(key=lambda x: x.objective_values[i])
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            max_objective = front[-1].objective_values[i]
            min_objective = front[0].objective_values[i]
            if not np.isclose(min_objective, max_objective):
                for j in range(1, len(front) - 1):
                    front[j].crowding_distance += (front[j + 1].objective_values[i] - front[j - 1].objective_values[i])\
                                                  / (max_objective - min_objective)
                    j += 1
            i += 1
            # TODO: Sorting by objective values: will not work in constrained problem


class PopIndividual(AbstractPopIndividual):
    """represents an individual in a population for NSGA-II"""

    def __init__(self, d_vars, objectives, constraints=None, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        super().__init__(d_vars, objectives, constraints, objective_values, total_constraint_violation)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def __str__(self):
        """string representation of the individual"""
        s = "solution at {}, rank {}".format(self.d_vars, self.rank)
        return s

    def __repr__(self):
        """representation of the individual"""
        return "[{:.2f}, {:.2f}], cd={:.2f}".format(self.d_vars[0], self.d_vars[1], self.crowding_distance)

    def __lt__(self, other):
        return ((self.rank < other.rank) or (
                (self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def clear(self):
        self.domination_count = 0
        self.dominated_set = []
        self.crowding_distance = 0


class Tests:

    # ------------------------tests for PopIndividual-------------------------
    @staticmethod
    def test__lt__():
        objectives = [lambda x: x[0]]
        x = PopIndividual([1], objectives)
        y = PopIndividual([2], objectives)
        z = PopIndividual([3], objectives)
        q = PopIndividual([4], objectives)
        r = PopIndividual([5], objectives)
        t = PopIndividual([6], objectives)
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
        assert x < y is True
        assert x < z is False
        assert x < x is False
        assert q < r is True
        assert q < t is False

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


if __name__ == "__main__":
    Tests.test_dominates()
    Tests.test_compare()
    Tests.test_recombine()
    Tests.test_mutate_individual()
    Tests.test_non_dominated_sort()
    Tests.test_crowding_distance_assignment()
    Tests.test_crowded_sort()
    Tests.test_tournament_selection()
    Tests.test_crossover()
    Tests.test_mutate()

