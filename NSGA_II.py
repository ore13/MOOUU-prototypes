"""
NSGA-II algorithim for multi objective optimisation
benchmark testing for SAM MOOUU using Differential evolution
version proposed by Deb[2002] IEEE
GNS.cri
Otis Rea
"""
import numpy as np
from Abstract_Moo import *
import math


class NSGA_II(AbstractMOEA):
    """implements the NSGA_II algorithm for multi objective optimisation
       Methods to be used:
       __init__: initialise the algorithm.
            Objectives: a list of functions to be optimised, e.g [f1, f2...]
            bounds: limits on the decision variables in the problem, to be specified as
            ([x1_lower, x1_upper], [x2_lower, x2_upper], ...)
            Other values are just numbers specifying parameters of the algorithm
        run: runs the algorithm and returns an approximation to the pareto front
    """

    # ------------------------External methods--------------------------------
    def __init__(self, objectives, bounds, parent_pop_size=100, cross_prob=0.9, cross_dist=15, mut_prob=0.01,
                 mut_dist=20, iterations=20):
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
        super().__init__(objectives, bounds, cross_prob, cross_dist, mut_prob, mut_dist, iterations)
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
        self.new_child_population()
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
            self.new_child_population()
        return self.fronts[0]

    def new_child_population(self):
        """Internal method. creates a new child population from the parent using tournament selection, crossover,
            and mutation."""
        self.child_pop = []
        self.child_pop = self.tournament_selection(self.parent_pop, self.parent_pop_size)
        self.crossover_step_SBX(self.child_pop)
        self.mutation_step_polynomial(self.child_pop)

    def non_dominated_sort(self, init=False):
        """Internal method. Sorts the population into a set of non-dominated pareto fronts F1, F2 ..."""
        if init:
            self.population = self.parent_pop
        for p in self.population:
            p.clear()
        self.fronts = []
        Q = set()
        for i in range(len(self.population)):
            p = self.population[i]
            for j in range(i + 1, len(self.population)):
                q = self.population[j]
                if p.dominates(q):
                    p.dominated_set.append(q)
                    q.domination_count += 1
                elif q.dominates(p):
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
        n = len(front)
        for individual in front:
            individual.crowding_distance = 0
        for f in self.objectives:
            sort_key = lambda x: f(x.values)
            front.sort(key=sort_key)
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            fmax = f(front[-1].values)
            fmin = f(front[0].values)
            if not np.isclose(fmax, fmin):
                for i in range(1, n - 1):
                    front[i].crowding_distance = front[i].crowding_distance + (
                                f(front[i + 1].values) - f(front[i - 1].values)) / (fmax - fmin)


class PopIndividual(AbstractPopIndividual):
    """represents an individual in a population for NSGA-II"""

    def __init__(self, values, objectives):
        """initialise the new population member"""
        super().__init__(values, objectives)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def __str__(self):
        """string representation of the individual"""
        s = "solution at {}, rank {}".format(self.values, self.rank)
        return s

    def __repr__(self):
        """representation of the individual"""
        return "[{:.2f}, {:.2f}], cd={:.2f}".format(self.values[0], self.values[1], self.crowding_distance)

    def __lt__(self, other):
        return ((self.rank < other.rank) or (
                (self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def clear(self):
        self.domination_count = 0
        self.dominated_set = []
