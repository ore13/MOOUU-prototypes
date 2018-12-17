"""abstract class for MOEA and PopIndividual classes"""
import numpy as np
import itertools as itr


class AbstractMOEA:
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
                 constraints=None, cross_prob=0.8, cross_dist=20, mut_prob=0.01, mut_dist=20, iterations=20):
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
        self.constraints = constraints
        self.number_objectives = number_objectives
        self.is_constrained = constraints is not None
        self.objectives = objectives
        self.bounds = bounds
        self.cross_prob = cross_prob
        self.cross_dist = cross_dist
        self.mut_prob = mut_prob
        self.mut_dist = mut_dist
        self.iterations = iterations
        self.animator_points = []

    def run(self):
        pass

    @staticmethod
    def tournament_selection(old_population, new_max_size, is_better=lambda x, y: x.fitness < y.fitness):
        new_population = []
        for _ in range(2):
            np.random.shuffle(old_population)
            i = 0
            while i < (new_max_size // 2):
                if is_better(old_population[2 * i], old_population[2 * i + 1]):  # Change this to a method -> __lt__ no more
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
            d_vars = []
            for j in range(len(self.bounds)):
                d_vars.append(self.bounds[j][0] + np.random.random() * (self.bounds[j][1] - self.bounds[j][0]))
            new_population.append(population_class(d_vars, self.objectives, self.constraints))
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

    def dominates(self, a, b):
        if self.is_constrained:
            return a.constrain_dominates(b)
        else:
            return a.dominates(b)

    def covers(self, a, b):
        if self.is_constrained:
            return a.constrain_covers(b)
        else:
            return a.covers(b)

    def reset_population(self, population):
        gen = []
        for _ in range(self.number_objectives):
            gen.append(np.zeros(len(population)))
        count = 0
        for individual in population:
            for i, obj in enumerate(individual.objective_values):
                gen[i][count] = obj
            individual.clear()
            count += 1
        self.animator_points.append(gen)

    def get_animation_points(self):
        return self.animator_points


class AbstractPopIndividual:
    """represents an individual in a population for NSGA-II"""

    def __init__(self, d_vars, objectives, constraints=None, objective_values=None, total_constraint_violation=None):
        """initialise the new population member"""
        self.d_vars = np.array(d_vars, dtype=float)
        self.objectives = objectives
        self.fitness = 0
        self.is_constrained = not (constraints is None)
        if objective_values is None:
            self.objective_values = self.calculate_objective_values()
        else:
            self.objective_values = objective_values
        if self.is_constrained:
            self.constraints = constraints
            if total_constraint_violation is None:
                self.total_constraint_violation = self.calculate_constrained_values(self.constraints, self.d_vars)
            else:
                self.total_constraint_violation = total_constraint_violation
            self.violates = self.total_constraint_violation > 0

    def update(self):
        if not self.is_constrained:
            self.objective_values = self.calculate_objective_values()
        else:
            self.total_constraint_violation = self.calculate_constrained_values(self.constraints, self.d_vars)
            self.violates = self.total_constraint_violation > 0
            if not self.violates: # if it does not violate, THEN calculate objective values in the constrained case
                self.objective_values = self.calculate_objective_values()

    def __str__(self):
        return "d_vars: {}, objectives: {}".format(self.d_vars, self.objective_values)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return bool(self.fitness < other.fitness)

    def __le__(self, other):
        return bool(self.fitness <= other.fitness)

    def __gt__(self, other):
        return bool(self.fitness > other.fitness)

    def __ge__(self, other):
        return bool(self.fitness >= other.fitness)

    def fitness_equals(self, other):
        return bool(self.fitness == other.fitness)

    def calculate_objective_values(self):
        # obj_vector = np.zeros(len(self.objectives), dtype=float)
        # for i, objective in enumerate(self.objectives):
        #     obj_vector[i] = objective(self.d_vars)
        return self.objectives(self.d_vars)

    def constrain_dominates(self, other):
        if self.violates and other.violates:
            result = bool(self.total_constraint_violation < other.total_constraint_violation)
        elif self.violates:
            result = False
        elif other.violates:
            result = True
        else:
            result = self.dominates(other)
        return result

    def constrain_covers(self, other):
        if self.violates and other.violates:
            result = self.total_constraint_violation < other.total_constraint_violation
        elif self.violates:
            result = False
        elif other.violates:
            result = True
        else:
            result = self.covers(other)
        return result

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
        if self.is_constrained and self.violates:
            return cls(self.d_vars, self.objectives, self.constraints,
                       total_constraint_violation=self.total_constraint_violation)
        elif self.is_constrained:
            return cls(self.d_vars, self.objectives, self.constraints, self.objective_values,
                       self.total_constraint_violation)
        else:
            return cls(self.d_vars, self.objectives, objective_values=self.objective_values)

    def crossover_SBX(self, other, bounds, distribution_parameter):
        """uses simulated binary crossover"""
        for i in range(len(self.d_vars)):
            if np.random.random() >= 0.5:
                p1 = self.d_vars[i]
                p2 = other.d_vars[i]
                if np.isclose(p1, p2, rtol=0, atol=1e-15):
                    beta_1, beta_2 = self.get_beta(np.NaN, np.NaN, distribution_parameter, values_are_close=True)
                else:
                    lower_transformation = (p1 + p2 - 2 * bounds[i][0]) / (abs(p2 - p1))
                    upper_transformation = (2 * bounds[i][1] - p1 - p2) / (abs(p2 - p1))
                    beta_1, beta_2 = self.get_beta(lower_transformation, upper_transformation, distribution_parameter)
                self.d_vars[i] = 0.5 * ((p1 + p2) - beta_1 * abs(p2 - p1))
                other.d_vars[i] = 0.5 * ((p1 + p2) + beta_2 * abs(p2 - p1))
        self.update()
        other.update()

    @staticmethod
    def get_beta(transformation1, transformation2, distribution_parameter, values_are_close=False):
        rand = np.random.random()
        beta_values = []
        for transformation in [transformation1, transformation2]:
            if values_are_close:
                p = 1
            else:
                p = 1 - 1/(2 * transformation ** (distribution_parameter + 1))
            u = rand * p
            if u <= 0.5:
                beta_values.append((2 * u) ** (1 / (distribution_parameter + 1)))
            else:
                beta_values.append((1 / (2 - 2 * u)) ** (1 / (distribution_parameter + 1)))
        return beta_values

    def mutate_polynomial(self, variable, bounds, distribution_parameter):
        u = np.random.random()
        if u <= 0.5:
            delta = (2 * u) ** (1 / (distribution_parameter + 1)) - 1
            self.d_vars[variable] = self.d_vars[variable] + delta * (self.d_vars[variable] - bounds[variable][0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (1 + distribution_parameter))
            self.d_vars[variable] = self.d_vars[variable] + delta * (bounds[variable][1] - self.d_vars[variable])
        self.update()

    @staticmethod
    def calculate_constrained_values(constraints, d_vars):
        """return the total constraint violation. Assumes all constraints are of the form g(X) >= 0"""
        total_constraint_violation = 0
        for constraint in constraints:
            g = constraint(d_vars)
            if g < 0:
                total_constraint_violation += abs(g)
        return total_constraint_violation

    def clear(self):
        self.fitness = 0


class Tests:

    @staticmethod
    def test__init__():
        objectives = [lambda x: x[0], lambda x: (1 + x[1])/x[0]]
        constraints = [lambda x: x[1] + 9 * x[0] - 6, lambda x: -x[1] + 9 * x[0] - 1]
        a = AbstractPopIndividual([0.2, 1], objectives)
        assert a.fitness == 0
        assert np.all(np.isclose(a.d_vars, np.array([0.2, 1])))
        assert a.objectives == objectives
        assert a.is_constrained is False
        assert np.all(np.isclose(a.objective_values, [0.2, 10]))
        b = AbstractPopIndividual([0.2, 1], objectives, objective_values=np.array([0.2, 10], dtype=float))
        assert b.is_constrained is False
        assert np.all(np.isclose(b.objective_values, [0.2, 10]))
        c = AbstractPopIndividual([0.2, 1], objectives, constraints)
        assert c.is_constrained
        assert c.constraints == constraints
        assert c.violates
        assert np.isclose(c.total_constraint_violation, 3.4)
        try:
            _ = c.objective_values
            raise Exception("objective values should not be defined")
        except AttributeError:
            pass
        d = AbstractPopIndividual([0.60000001, 0.6], objectives, constraints)
        assert d.violates is False
        assert np.isclose(d.total_constraint_violation, 0)
        assert np.all(np.isclose(d.objective_values, [0.6, 8/3]))

    @staticmethod
    def test_update():
        objectives = [lambda x: x[0], lambda x: (1 + x[1]) / x[0]]
        constraints = [lambda x: x[1] + 9 * x[0] - 6, lambda x: -x[1] + 9 * x[0] - 1]
        a = AbstractPopIndividual([0.3, 1], objectives, constraints)
        a.d_vars = np.array([0.2, 1], dtype=float)
        a.update()
        assert a.is_constrained
        assert a.constraints == constraints
        assert a.violates
        assert np.isclose(a.total_constraint_violation, 3.4)
        a.d_vars = np.array([0.6000001, 0.6])
        a.update()
        assert a.violates is False
        assert np.isclose(a.total_constraint_violation, 0)
        assert np.all(np.isclose(a.objective_values, [0.6, 8/3]))
        b = AbstractPopIndividual([0.1, 1], objectives)
        b.d_vars = np.array([0.2, 1], dtype=float)
        b.update()
        assert b.fitness == 0
        assert np.all(np.isclose(b.d_vars, np.array([0.2, 1])))
        assert b.objectives == objectives
        assert b.is_constrained is False
        assert np.all(np.isclose(b.objective_values, [0.2, 10]))

    @staticmethod
    def test_dominates():
        # test very simple objective function
        objectives = [lambda x: x[0]]
        population = []
        for i in range(5):
            population.append(AbstractPopIndividual([i], objectives))
        assert population[0].dominates(population[1]) is True
        assert population[0].dominates(population[3]) is True
        assert population[3].dominates(population[0]) is False
        assert population[0].dominates(population[0]) is False

    @staticmethod
    def test_SBX():
        np.random.seed(1002402)
        bounds = [(-np.inf, np.inf)]
        objectives = [lambda x: x[0]]
        dist_param = 3
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.022211826787321, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.977788173212678, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.0, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 10.0, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.0, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 10.0, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.998068382956265, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.001931617043736, atol=1e-6)
        p1 = AbstractPopIndividual([5], objectives)
        p2 = AbstractPopIndividual([10], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 5.746398124977722, atol=1e-6)
        assert np.isclose(p2.d_vars[0], 9.25360187502228, atol=1e-6)
        np.random.seed(100)
        bounds = [(0, 5)]
        p1 = AbstractPopIndividual([3], objectives)
        p2 = AbstractPopIndividual([4], objectives)
        p1.crossover_SBX(p2, bounds, dist_param)
        assert np.isclose(p1.d_vars[0], 3.068122818593165)
        assert np.isclose(p2.d_vars[0], 3.9312316083035745)

    @staticmethod
    def test_mutate_polynomial():
        np.random.seed(12645678)
        bounds = [(-5, 5)]
        distribution_param = 2
        objectives = [lambda x: x[0]]
        d_var = 0
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.07299526594720773)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.027816685123562834)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], -0.9019673485855295)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 0.49704076190606683)
        np.random.seed(12645678)
        d_var = 4.9
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.755469373424529)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.844922963455346)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 3.1141046498006517)
        p = AbstractPopIndividual([d_var], objectives)
        p.mutate_polynomial(0, bounds, distribution_param)
        assert np.isclose(p.d_vars[0], 4.909940815238122)

    @staticmethod
    def test_constrain_dominates():
        objectives = [lambda x: x[0], lambda x: (1 + x[1]) / x[0]]
        constraints = [lambda x: x[1] + 9 * x[0] - 6, lambda x: -x[1] + 9 * x[0] - 1]
        a = AbstractPopIndividual([0.6, 0.7], objectives, constraints)
        b = AbstractPopIndividual([0.8, 2], objectives, constraints)
        c = AbstractPopIndividual([0.2, 1], objectives, constraints)
        d = AbstractPopIndividual([0.1, 3], objectives, constraints)
        assert a.constrain_dominates(b)
        assert a.constrain_dominates(c)
        assert a.constrain_dominates(d)
        assert b.constrain_dominates(c)
        assert b.constrain_dominates(d)
        assert c.constrain_dominates(d)
        assert b.constrain_dominates(a) is False
        assert c.constrain_dominates(a) is False
        assert d.constrain_dominates(a) is False
        assert c.constrain_dominates(b) is False
        assert d.constrain_dominates(b) is False
        assert d.constrain_dominates(c) is False

    @staticmethod
    def test_constrain_covers():
        pass


if __name__ == "__main__":
    Tests.test__init__()
    Tests.test_update()
    Tests.test_dominates()
    Tests.test_SBX()
    Tests.test_mutate_polynomial()
    Tests.test_constrain_dominates()

