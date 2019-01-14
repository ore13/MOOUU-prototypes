
from Wrappers.StochasticIO import *
from TestProblems.StochasticProblemSuite import *

test_functions = {"stochasticparaboloid": Stp.StochasticParaboloid, "stochasticparaboloid2": Stp.StochasticParaboloid2}


def generate_file(filename, model):
    if os.path.exists(filename):
        os.remove(filename)
    index = ['d_var{}'.format(i + 1) for i in range(model.number_decision_variables())]
    index += ['par{}'.format(i + 1) for i in range(model.number_parameters())]
    data = generate_random(model)
    df = pd.Series(data, index, name='MODEL INPUT FILE')
    df.to_csv(filename, header=True, encoding='ascii')


def generate_random(model):
    d_vars = np.random.random(model.number_decision_variables())
    min_bound = np.array([min(bound) for bound in model.bounds()])
    max_bound = np.array([max(bound) for bound in model.bounds()])
    d_vars = min_bound + d_vars * (max_bound - min_bound)
    pars = stat.multivariate_normal.rvs(model.parameter_means(), model.parameter_covariance())
    if type(pars) != np.ndarray:
        pars = np.array([pars])
    return np.concatenate((d_vars, pars))


def test():
    model_name = 'stochasticparaboloid'
    model = test_functions[model_name]
    input_file = "input.dat"
    generate_file(input_file, model)
    path = os.getcwd()
    path = os.path.join(path, 'Wrappers', 'StochasticIO.py')
    subprocess.run(['python', path, model_name])


if __name__ == "__main__":
    test()
