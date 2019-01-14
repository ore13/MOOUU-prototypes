"""SPEA_2 algorithm"""

from GeneticAlgorithms.Abstract_Moo import *
import itertools as itr


class SPEA_2(AbstractMOEA):
    """Represents the SPEA_2 algorithm"""

    def __init__(self, objectives, bounds, number_objectives, constraints=None, population_size=100, archive_size=100,
                 cross_prob=0.9, cross_dist=15, mut_prob=0.01, mut_dist=20, iterations=20):
        super().__init__(objectives, bounds, number_objectives, constraints=constraints, cross_prob=cross_prob,
                         cross_dist=cross_dist, mut_prob=mut_prob, mut_dist=mut_dist, iterations=iterations)
        self.population_size = population_size
        self.archive_size = archive_size
        self.population = []
        self.archive = []
        self.next_archive = []
        self.joint = []
        self.distance_graph = dict()

    def run(self):
        self.population = self.initialise_population(self.population_size, PopIndividual)
        self.run_model(self.population)
        self.archive = []
        for _ in range(self.iterations):
            self.joint = self.population + self.archive
            self.fitness_assignment()  # Assigns fitness d_vars to all population members
            self.environmental_selection()  # create new archive from current archive and population
            self.population = self.tournament_selection(self.archive, self.population_size)
            self.crossover_step_SBX(self.population)
            self.mutation_step_polynomial(self.population)
            self.run_model(self.population)
            self.reset_population(self.archive)
        return self.archive

    def __str__(self):
        return "SPEA-2"

    def fitness_assignment(self):
        k = int(np.sqrt(self.population_size + self.archive_size)) - 1
        strength_dict = dict()
        for i, individual1 in enumerate(self.joint):
            distances = np.zeros(len(self.joint) - 1)
            for j, individual2 in itr.filterfalse(lambda x: x[0] == i, enumerate(self.joint)):
                if j > i:
                    j -= 1
                if individual1.dominates(individual2): #self.dominates(individual1,individual2):
                    strength_dict[individual1] = strength_dict.get(individual1, 0) + 1
                    individual2.dominating_set.append(individual1)
                distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
                distances[j] = distance
            individual1.fitness_distances = distances
        for individual1 in self.joint:
            raw_fitness = 0
            for individual2 in individual1.dominating_set:
                raw_fitness += strength_dict.get(individual2, 0)
            individual1.fitness_distances.partition(k)
            density = individual1.fitness_distances[k]
            density = 1 / (density + 2)
            individual1.fitness = raw_fitness + density
        # TODO: fix use of objective_values...

    def environmental_selection(self):
        self.next_archive = self.get_non_dominated()
        distances_initialised = False
        while len(self.next_archive) > self.archive_size:
            if not distances_initialised:
                self.distance_assignment()
                distances_initialised = True
            self.distance_sort()
            self.truncate()
        while len(self.next_archive) < self.archive_size:
            k = self.archive_size - len(self.next_archive)
            a = np.partition(self.joint, k - 1)[:k]
            self.next_archive += list(a)
        self.archive = self.next_archive

    def get_non_dominated(self):
        mark = 0
        for i in range(len(self.joint)):
            if self.joint[i].fitness <= 1:
                if mark < i:
                    self.joint[i], self.joint[mark] = self.joint[mark], self.joint[i]
                mark += 1
        next_archive, self.joint = self.joint[:mark], self.joint[mark:]
        return next_archive

    def distance_assignment(self):
        for individual1, individual2 in itr.combinations(self.next_archive, 2):
            distance = np.linalg.norm(individual1.objective_values - individual2.objective_values, 2)
            individual1.connect(individual2, distance)
        # TODO: objective_values...

    def truncate(self):
        i = 0
        while i < len(self.next_archive):
            individual1 = self.next_archive[i]
            min_individual = True
            j = 0
            while min_individual and j < len(self.next_archive):
                min_individual = individual1.less_than(self.next_archive[j])
                j += 1
            if min_individual:
                self.next_archive.pop(i)
                i = len(self.next_archive)
                individual1.delete()
            i += 1

    def distance_sort(self):
        for individual in self.next_archive:
            individual.sort()


class PopIndividual(AbstractPopIndividual):
    """PopIndividual for SPEA algoritm"""

    def __init__(self, d_vars, constraints=None, objective_values=None, total_constraint_violation=None):
        super().__init__(d_vars, constraints, objective_values, total_constraint_violation)
        self.dominating_set = []
        self.fitness_distances = None
        self.keys = []
        self.distances = dict()

    def __str__(self):
        return "d_vars: {}, objectives: {}".format(self.d_vars, self.objective_values)

    def __repr__(self):
        return str(self)

    def connect(self, other, distance):
        other.keys.append(self)
        other.distances[self] = distance
        self.keys.append(other)
        self.distances[other] = distance

    def disconnect(self, other):
        self.distances.pop(other)
        other.distances.pop(self)

    def delete(self):
        for other in self.keys:
            self.disconnect(other)

    def sort(self):
        self.keys.sort(key=lambda x: self.distances.get(x, np.inf))
        i = -1
        while abs(i) <= len(self.keys) and self.distances.get(self.keys[i]) is None:
            self.keys.pop(i)
            i -= 1

    def less_than(self, other):
        eq = True
        lt = False
        i = 0
        while i < len(self.keys) and not lt and eq:
            eq = self.distances[self.keys[i]] <= other.distances[other.keys[i]]
            lt = self.distances[self.keys[i]] < other.distances[other.keys[i]]
            i += 1
        return eq or lt

    def clear(self):
        self.keys = []
        self.distances = dict()
        self.dominating_set = []
        self.fitness_distances = None
