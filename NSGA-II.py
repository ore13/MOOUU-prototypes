
# NSGA-II algorithim for multi objective optimisation
# benchmark testing for SAM MOOUU using Differential evolution
# version proposed by Deb[2002] IEEE
# Otis Rea - GNS cri
import numpy as np
import math

class NSGA_II:
    """NSGA-II algorithm"""

    def __init__(self, objectives, constraints, pop_size):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions
           Constraints: array of constraints (To be worked on)"""
        self.objectives = objectives
        self.constraints = constraints
        self.population = self.initialise_pop()
        self.fronts = []
        self.pareto_optimal_set = []
        self.parent_pop = []
        self.child_pop = []
        self.pop_size = pop_size

    def run(self):
        """run NSGA-II algorithm after initialisation"""
        #Do some setup stuff
        self.setup()
        self.main_algorithm()
        return self.pareto_optimal_set

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

    def setup(self):
        pass

    def non_dominated_sort(self):
        """non dominated sort used to find the set of pareto fronts F1, F2, ... etc in the
        NSGA-II algorithim. P must be represented as a array of PopIndividuals
        Algorithim proposed by Deb[2002]"""
        self.fronts.append([])
        for i in range(0, len(self.population)):
            p = self.population[i]
            for j in range(0, len(self.population)):
                q = self.population[j]
                if self.dominates(p, q):
                    p.dominated_set.append(q)
                elif self.dominates(q, p):
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
        front = self.fronts[front_index]
        n = len(front)
        for f in self.objectives:
            sort_key = lambda individual: f(individual.value)
            front.sort(key=sort_key)
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            fmax = f(front[-1].value)
            fmin = f(front[0].value)
            for i in range(1, n - 1):
                front[i].crowding_distance = front[i].crowding_distance + ( f(front[i+1].value) - f(front[i-1].value) )\
                    / (fmax - fmin)

    def initialise_pop(self):
        """create set of decision variables based on the constraints"""
        return []

    def dominates(self, x1, x2):
        i = 0
        weak_dom = True
        strong_condition = False
        while i < len(self.objectives):
            if self.objectives[i](x1.value) > self.objectives[i](x2.value):
                weak_dom = False
                i = len(self.objectives)
            elif self.objectives[i](x1.value) < self.objectives[i](x2.value):
                strong_condition = True
            i += 1
        return weak_dom and strong_condition

    # ----------------------front sorting algorithms--------------------------
    def crowded_sort(self, front_index):
        """implements an insertion sort using the crowded comparison operator"""
        for i in range(1, len(self.fronts[front_index])):
            value = self.fronts[front_index][i]
            j = i - 1
            in_place = False
            while j > 0 and not in_place:
                if self.crowded_comparison(value, self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                else:
                    self.fronts[front_index][j + 1] = value
                    in_place = True
                j -= 1
            if j == 0 and not in_place:
                if self.crowded_comparison(value, self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                    self.fronts[front_index][j] = value
                else:
                    self.fronts[front_index][j + 1] = value

    @staticmethod
    def crowded_comparison(indiv_1, indiv_2):
        """returns true if indiv_1 is determined better through crowded comparison.
        returns False if indiv_2 is greater by crowded comparison"""
        return (indiv_1.rank < indiv_2.rank) or ((indiv_1.rank == indiv_2.rank)
                                           and (indiv_1.crowding_distance > indiv_2.crowding_distance))

#--------------------------Other Classes--------------------------------------
class PopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, value):
        self.value = value
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def __str__(self):
        s = "solution at {}, rank {}".format(self.value, self.rank)
        return s

    def __repr__(self):
        return str(self.value)

#----------------------------Tests-----------------------
class Tests:

    @staticmethod
    def test_dominates():
        #test very simple objective function
        F = [lambda x: x[0]]
        bounds = (-2, 2)
        MOO = NSGA_II(F, bounds, 20)
        for i in range(5):
            MOO.population.append(PopIndividual([i]))
        assert MOO.dominates(MOO.population[0], MOO.population[1]) is True
        assert MOO.dominates(MOO.population[0], MOO.population[3]) is True
        assert MOO.dominates(MOO.population[3], MOO.population[0]) is False
        assert MOO.dominates(MOO.population[0], MOO.population[0]) is False

        #test more complex objective function
        #F = [lambda x: ]
        print("Passed")

    @staticmethod
    def test_non_dominated_sort():
        #test very simple objective function
        F = [lambda x: x[0]]
        bounds = (-2, 2)
        MOO = NSGA_II(F, bounds, 20)
        for i in range(5):
            MOO.population.append(PopIndividual([i]))
        MOO.non_dominated_sort()
        print(MOO.fronts)

        #test more complex objective function
        F = [lambda x: x[0], lambda x: 1 / x[0] + x[1]]
        MOO = NSGA_II(F, bounds, 20)
        for i in range(1, 5):
            for j in range(1, 5):
                MOO.population.append(PopIndividual([i, j]))
        MOO.non_dominated_sort()
        print(MOO.fronts)

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
        assert math.isclose(b.crowding_distance, 14/9)
        print("passed tests")

    @staticmethod
    def test_crowded_sort():
        F = []
        bounds = []
        MOO = NSGA_II(F, bounds, 20)
        front = []
        for i in range(5):
            front.append(PopIndividual([i]))
            front[i].rank = i
        MOO.fronts.append(front)
        MOO.crowded_sort(0)
        assert MOO.fronts[0] = []


def main():
    Tests.test_dominates()
    Tests.test_non_dominated_sort()
    Tests.test_crowding_distance_assignment()

main()
