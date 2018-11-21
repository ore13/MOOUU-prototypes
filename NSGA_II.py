# NSGA-II algorithim for multi objective optimisation
# benchmark testing for SAM MOOUU using Differential evolution
# version proposed by Deb[2002] IEEE
# GNS.cri
import numpy as np
import math


class NSGA_II:
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
    def __init__(self, objectives, bounds, parent_pop_size=50, cross_prob=0.5, cross_dist=20, mut_prob=0.01,
                 mut_dist=20, iterations=30):
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
        self.objectives = objectives
        self.bounds = bounds
        self.parent_pop_size = parent_pop_size
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        self.iterations = iterations
        self.n_decision_variables = len(bounds)
        # visualisation storage variables
        self.population_storage = []
        # --------------------------------sets of population members------------------------------------------
        self.parent_pop = []
        self.child_pop = []
        self.fronts = []
        self.population = []

    def run(self):
        """Run the NSGA-II algorithm. This will return an approximation to the pareto front"""
        self.initialise_parent_population()
        self.non_dominated_sort(init=True)
        self.new_child_population()
        for i in range(self.iterations):
            self.population = self.child_pop + self.parent_pop
            self.population_storage.append(self.population)
            self.non_dominated_sort()
            self.parent_pop = []
            j = 0
            while len(self.parent_pop) + len(self.fronts[j]) < self.parent_pop_size:
                self.crowding_distance_assignment(j)
                self.parent_pop += self.fronts[j]
                j += 1
            self.crowding_distance_assignment(j)
            self.crowded_sort(j)
            self.parent_pop += self.fronts[j][:(self.parent_pop_size - len(self.parent_pop))]
            self.new_child_population()
        return self.fronts[0]

    def new_child_population(self):
        """Internal method. creates a new child population from the parent using tournament selection, crossover,
            and mutation."""
        self.child_pop = []
        self.tournament_selection()
        self.crossover()
        self.mutate()

    def initialise_parent_population(self):
        """Internal method. Create the initial parent population"""
        for i in range(self.parent_pop_size):
            values = []
            for j in range(self.n_decision_variables):
                values.append(self.bounds[j][0] + np.random.random() * (self.bounds[j][1] - self.bounds[j][0]))
            self.parent_pop.append(PopIndividual(values))

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
                if p.dominates(q, self.objectives):
                    p.dominated_set.append(q)
                    q.domination_count += 1
                elif q.dominates(p, self.objectives):
                    q.dominated_set.append(p)
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                Q.add(p)
        self.fronts.append(list(Q))
        i = 0
        while self.fronts[i] != []:
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
            for i in range(1, n - 1):
                front[i].crowding_distance = front[i].crowding_distance + (
                            f(front[i + 1].values) - f(front[i - 1].values)) / (fmax - fmin)

    def crowded_sort(self, front_index):
        """Internal method. Uses an insertion sort with a self defined comparison operator to sort the front into
            order with the best individuals at the front"""
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
        """Internal method. performs a tournament selection on the parent population. Note that it is assumed that the
            population has already been ranked and been assigned crowding distances"""
        for _ in range(2):
            np.random.shuffle(self.parent_pop)
            i = 0
            while i < (self.parent_pop_size // 2):
                if self.parent_pop[2 * i].compare(self.parent_pop[2 * i + 1]):
                    self.child_pop.append(self.parent_pop[2 * i].clone())
                else:
                    self.child_pop.append(self.parent_pop[2 * i + 1].clone())
                i += 1
        if self.parent_pop_size % 2 == 1:  # i.e is odd
            self.child_pop.append(self.parent_pop[self.parent_pop_size - 1].clone())
        # child population now populated from tournament selection

    def crossover(self):
        """Internal method. Crosses a random selection of the child population"""
        for i in range(len(self.child_pop)):
            if np.random.random() > self.cross_prob:
                rand_index = np.random.randint(0, len(self.child_pop))
                self.child_pop[i].recombine(self.child_pop[rand_index], self.bounds, self.cross_dist)

    def mutate(self):
        """Internal method. Use mutation clock to mutate a small selection of the child population"""
        k = 0
        i = 0
        while i < self.parent_pop_size:
            self.child_pop[i].mutate_individual(k, self.bounds, self.mut_dist)
            length = int(np.ceil(- 1 / self.mut_prob * np.log(1 - np.random.random())))
            i += (k + length) // self.n_decision_variables
            k = (k + length) % self.n_decision_variables


class PopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, values):
        """initialise the new population member"""
        self.values = np.array(values, dtype=float)
        # self.byte_encoding = self._encode(value)
        self.dominated_set = []
        self.domination_count = 0
        self.rank = None
        self.crowding_distance = 0

    def dominates(self, other, objectives):
        """tests for dominance between self and other. returns true if self dominates other, false otherwise"""
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
        """string representation of the individual"""
        s = "solution at {}, rank {}".format(self.values, self.rank)
        return s

    def __repr__(self):
        """representation of the individual"""
        return "[{:.2f}, {:.2f}], cd={:.2f}".format(self.values[0], self.values[1], self.crowding_distance)

    def compare(self, other):
        """compares self and other in terms of their rank and crowding distance. Returns true the rank of self is lower
            than other, or if the ranks are equal and the crowding distance of self is higher."""
        return ((self.rank < other.rank) or (
                    (self.rank == other.rank) and (self.crowding_distance > other.crowding_distance)))

    def recombine(self, other, bounds, distribution_parameter):
        """uses simulated binary crossover"""
        for i in range(len(self.values)):
            if np.random.random() >= 0.5:
                p1 = self.values[i]
                p2 = other.values[i]
                if np.isclose(p1, p2):
                    B1 = self.get_beta(0, distribution_parameter, is_close_values=True)
                    B2 = self.get_beta(0, distribution_parameter, is_close_values=True)
                else:
                    beta_l = (p1 + p2 - 2 * bounds[i][0]) / (abs(p2 - p1))
                    beta_u = (2 * bounds[i][1] - p1 - p2) / (abs(p2 - p1))
                    B1 = self.get_beta(beta_l, distribution_parameter)
                    B2 = self.get_beta(beta_u, distribution_parameter)
                self.values[i] = 0.5 * ((p1 + p2) - B1 * abs(p2 - p1))
                other.values[i] = 0.5 * ((p1 + p2) + B2 * abs(p2 - p1))

    def clone(self):
        return PopIndividual(self.values)

    @staticmethod
    def get_beta(transformation, n, is_close_values=False):
        if is_close_values:
            p = 1
        else:
            p = 1 - 1 / (2 * transformation ** (n + 1))
        u = np.random.ranf() * p
        if u <= 0.5:
            beta = (2 * u) ** (1 / (n + 1))
        else:
            beta = (1 / (2 - 2 * u)) ** (1 / (n + 1))
        return beta

    def mutate_individual(self, variable, bounds, distribution_parameter):
        u = np.random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.values[variable] = self.values[variable] + delta * (self.values[variable] - bounds[variable][0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.values[variable] = self.values[variable] + delta * (bounds[variable][1] - self.values[variable])

    def clear(self):
        self.domination_count = 0
        self.dominated_set = []
