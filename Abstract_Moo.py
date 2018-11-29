"""abstract class for MOEA and PopIndividual classes"""
import numpy as np


class AbstractMOEA:
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
    def __init__(self, objectives, bounds, cross_prob=0.8, cross_dist=20, mut_prob=0.01,
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
        self.objectives = objectives
        self.bounds = bounds
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        self.iterations = iterations

    def run(self):
        pass

    @staticmethod
    def tournament_selection(old_population, new_max_size):
        new_population = []
        for _ in range(2):
            np.random.shuffle(old_population)
            i = 0
            while i < (new_max_size // 2):
                if old_population[2 * i] < old_population[2 * i + 1]:
                    new_population.append(old_population[2 * i].clone())
                else:
                    new_population.append(old_population[2 * i + 1].clone())
                i += 1
        if new_max_size % 2 == 1:  # i.e is odd
            new_population.append(old_population[-1].clone())
        return new_population

    def initialise_population(self, population_size, population_class):
        new_population = []
        for i in range(population_size):
            values = []
            for j in range(len(self.bounds)):
                values.append(self.bounds[j][0] + np.random.random() * (self.bounds[j][1] - self.bounds[j][0]))
            new_population.append(population_class(values, self.objectives))
        return new_population

    def crossover_step_SBX(self, population):
        for i in range(len(population)):
            if np.random.random() > self.cross_prob:
                rand_index = np.random.randint(0, len(population))
                population[i].crossover_SBX(population[rand_index], self.bounds, self.cross_dist)

    def mutation_step_polynomial(self, population):
        k = 0
        i = 0
        while i < len(population):
            population[i].mutate_polynomial(k, self.bounds, self.mut_dist)
            length = int(np.ceil(- 1 / self.mut_prob * np.log(1 - np.random.random())))
            i += (k + length) // len(self.bounds)
            k = (k + length) % len(self.bounds)

    def fast_non_dominated(self, population, key=lambda s, o: s.dominates(o)):
        """use Kung's algorithim to identify the non dominated set"""
        population.sort(key=lambda x: x.objective_values[0])
        return self._front(population, key=key)

    def _front(self, population, key):
        if len(population) == 1:
            return population
        else:
            mid = int(np.round(len(population) / 2))
            top = self._front(population[:mid], key)
            bottom = self._front(population[mid:], key)
            M = []
            for individual1 in bottom:
                dominated = False
                i = 0
                while not dominated and i < len(top):
                    if key(top[i], individual1):
                        dominated = True
                    i += 1
                if not dominated:
                    M.append(individual1)
            M += top
            return M


class AbstractPopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, values, objectives, objective_values=None):
        """initialise the new population member"""
        self.values = np.array(values, dtype=float)
        self.objectives = objectives
        if objective_values is None:
            self.objective_values = self.calculate_objective_values()
        else:
            self.objective_values = objective_values
        self.fitness = 0

    def __str__(self):
        return "values: {}, objectives: {}".format(self.values, self.objective_values)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def fitness_equals(self, other):
        return self.fitness == other.fitness

    def calculate_objective_values(self):
        obj_vector = np.zeros(len(self.objectives), dtype=float)
        for i, objective in enumerate(self.objectives):
            obj_vector[i] = objective(self.values)
        return obj_vector

    def dominates(self, other):
        """tests for dominance between self and other. returns true if self dominates other, false otherwise"""
        weak_dom = True
        strong_condition = False
        for i in range(len(self.objective_values)):
            if self.objective_values[i] > other.objective_values[i]:
                weak_dom = False
                i = len(self.objective_values)
            elif self.objective_values[i] < other.objective_values[i]:
                strong_condition = True
            i += 1
        return weak_dom and strong_condition

    def covers(self, other):
        condition = True
        for i in range(len(self.objective_values)):
            if self.objective_values[i] > other.objective_values[i]:
                condition = False
                i = len(self.objective_values)
            i += 1
        return condition

    def clone(self):
        cls = self.__class__
        return cls(self.values, self.objectives, objective_values=self.objective_values)

    def crossover_SBX(self, other, bounds, distribution_parameter):
        """uses simulated binary crossover"""
        for i in range(len(self.values)):
            if np.random.random() >= 0.5:
                p1 = self.values[i]
                p2 = other.values[i]
                if np.isclose(p1, p2, rtol=0, atol=1e-15):
                    B1 = self.get_beta(0, distribution_parameter, is_close_values=True)
                    B2 = self.get_beta(0, distribution_parameter, is_close_values=True)
                else:
                    beta_l = (p1 + p2 - 2 * bounds[i][0]) / (abs(p2 - p1))
                    beta_u = (2 * bounds[i][1] - p1 - p2) / (abs(p2 - p1))
                    B1 = self.get_beta(beta_l, distribution_parameter)
                    B2 = self.get_beta(beta_u, distribution_parameter)
                self.values[i] = 0.5 * ((p1 + p2) - B1 * abs(p2 - p1))
                other.values[i] = 0.5 * ((p1 + p2) + B2 * abs(p2 - p1))
                self.objective_values = self.calculate_objective_values()
                other.objective_values = other.calculate_objective_values()

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

    def mutate_polynomial(self, variable, bounds, distribution_parameter):
        u = np.random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.values[variable] = self.values[variable] + delta * (self.values[variable] - bounds[variable][0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.values[variable] = self.values[variable] + delta * (bounds[variable][1] - self.values[variable])
        self.objective_values = self.calculate_objective_values()
