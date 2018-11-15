
# NSGA-II algorithim for multi objective optimisation
# benchmark testing for SAM MOOUU using Differential evolution
# version proposed by Deb[2002] IEEE
# GNS.cri
import numpy as np
import math
#import byteManipulation as byt

class NSGA_II:
    """NSGA-II algorithm"""

    # ------------------------External methods--------------------------------
    def __init__(self, objectives, constraints, pop_size, cross_prob=0.5, mut=0.01):
        """initialise the algorithm. Parameters:
           objectives: vector of objective functions
           Constraints: array of constraints (To be worked on)"""
        self.cross_prob = cross_prob
        self.mut = mut
        self.objectives = objectives
        self.constraints = constraints
        self.population = []
        self.fronts = []
        self.pareto_optimal_set = []
        self.parent_pop = []
        self.child_pop = []
        self.pop_size = pop_size

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

    # ----------------------- internal methods-------------------------------
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
            sort_key = lambda individual: f(individual.value)
            front.sort(key=sort_key)
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            fmax = f(front[-1].value)
            fmin = f(front[0].value)
            for i in range(1, n - 1):
                front[i].crowding_distance = front[i].crowding_distance + ( f(front[i+1].value) - f(front[i-1].value) )\
                    / (fmax - fmin)

    def crowded_sort(self, front_index):
        """implements an insertion sort using the crowded comparison operator"""
        for i in range(1, len(self.fronts[front_index])):
            indiv_1 = self.fronts[front_index][i]
            j = i - 1
            in_place = False
            while j > 0 and not in_place:
                if indiv_1.crowded_comparison(self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                else:
                    self.fronts[front_index][j + 1] = indiv_1
                    in_place = True
                j -= 1
            if j == 0 and not in_place:
                if indiv_1.crowded_comparison(self.fronts[front_index][j]):
                    self.fronts[front_index][j + 1] = self.fronts[front_index][j]
                    self.fronts[front_index][j] = indiv_1
                else:
                    self.fronts[front_index][j + 1] = indiv_1

    def tournament_selection(self, crowded=False):
        """performs a tournament selection on the parent population. Note that it is assumed that the population has
        already been ranked and been assign"""
        np.random.shuffle(self.parent_pop)
        i = 0
        N = len(self.parent_pop)
        while i < N % 2:
            if self.parent_pop[2*i].is_better(self.parent_pop[2*i + 1]):
                self.child_pop.append(self.parent_pop[2*i])
            else:
                self.child_pop.append(self.parent_pop[2*i + 1])
            i += 1
        # now repeat this process again:
        i = 0
        np.random.shuffle(self.parent_pop)
        while i < N % 2:
            if self.parent_pop[2*i].is_better(self.parent_pop[2*i + 1]):
                self.child_pop.append(self.parent_pop[2*i])
            else:
                self.child_pop.append(self.parent_pop[2*i + 1])
            i += 1
        # now if length of parent_pop was odd, we have to add one more to have the same number of populations in the child and parent
        if N % 2 == 1: # i.e is odd
            self.child_pop.append(self.parent_pop[N - 1])
        # child population now populated from tourmenent selection

    def crossover(self):
        for i in range(len(self.child_pop)):
            if np.random.ranf() > self.cross_prob:
                rand_index = np.random.randint(0, len(self.child_pop))
                self.child_pop[i].recombine(self.child_pop[rand_index])

    def mutate(self):




#--------------------------Other Classes--------------------------------------
class PopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, values):
        self.values = values
        #self.byte_encoding = self._encode(value)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def dominates(self, other, objectives):
        i = 0
        weak_dom = True
        strong_condition = False
        while i < len(objectives):
            if objectives[i](self.values) > objectives[i](other.value):
                weak_dom = False
                i = len(objectives)
            elif objectives[i](self.values) < objectives[i](other.value):
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

    def is_better(self, other):
        return ((self.rank < other.rank) or ((self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def recombine(self, other):
        """cross self and other at a randomly selected point, by recombining hexadecimal representations"""
        rand_point = np.random.randint(0, len(self.values))
        temp = self.values[rand_point:]
        self.values[rand_point:] = other.values[rand_point:]
        other.values[rand_point:] = temp


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
        assert MOO.population[0].dominates(MOO.population[1], MOO.objectives) is True
        assert MOO.population[0].dominates(MOO.population[3], MOO.objectives) is True
        assert MOO.population[3].dominates(MOO.population[0], MOO.objectives) is False
        assert MOO.population[0].dominates(MOO.population[0], MOO.objectives) is False

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
        """unfinished"""
        F = []
        bounds = []
        MOO = NSGA_II(F, bounds, 20)
        front = []
        for i in range(5):
            front.append(PopIndividual([i]))
            front[i].rank = i
        MOO.fronts.append(front)
        MOO.crowded_sort(0)
        assert MOO.fronts[0] == []

    @staticmethod
    def test_recombine(self):
        pass

    @staticmethod
    def test_tournament_selec():
        pass




def main():
    Tests.test_dominates()
    Tests.test_non_dominated_sort()
    Tests.test_crowding_distance_assignment()

main()
