
import argparse
import numpy as np
import scipy.stats as stat
import os
import pandas as pd
import re
import subprocess
import time
import TestProblems.StochasticProblemSuite as Stp


test_functions = {"stochasticparaboloid": Stp.StochasticParaboloid, "stochasticparaboloid2": Stp.StochasticParaboloid2}


class IOWrapper:

    def __init__(self):
        args = self.parse()
        model = test_functions[args.benchmark_function.lower()]
        d_vars, pars = self.read_input_file(args.input_file, model)
        objectives = model.calculate_objectives(d_vars, pars)
        self.write_output_file(objectives, args.output_file)

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("benchmark_function", default='stochasticparaboloid')
        parser.add_argument("--input_file", dest='input_file', default='input.dat')
        parser.add_argument("--output_file", dest='output_file', default='output.dat')
        args = parser.parse_args()
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        if args.benchmark_function.lower() not in test_functions.keys():
            raise Exception("benchmark_function {} not found in known functions".format(args.benchmark_function))
        if not os.path.exists(args.input_file):
            raise Exception("input file not found")
        return args

    @staticmethod
    def read_input_file(input_file, model):
        df = pd.read_csv(input_file, encoding='ascii', squeeze=True, index_col=0)
        par_template = re.compile('par[0-9]+')
        d_var_template = re.compile('d_var[0-9]+')
        number_d_vars = 0
        number_pars = 0
        for i, index in enumerate(df.index):
            if d_var_template.fullmatch(index):
                number_d_vars += 1
            if par_template.fullmatch(index):
                number_pars += 1
        if number_d_vars != model.number_decision_variables():
            raise Exception("Incorrect number of decision variables for benchmark function")
        if number_pars != model.number_parameters():
            raise Exception("Incorrect number of parameters for benchmark function")
        data = df.values
        d_vars = data[:number_d_vars]
        pars = data[number_d_vars: number_d_vars + number_pars]
        return d_vars, pars

    @staticmethod
    def write_output_file(objectives, output_file):
        num_objectives = len(objectives)
        index = ['objective{}'.format(i + 1) for i in range(num_objectives)]
        df = pd.Series(objectives, index)
        f = open(output_file, 'w')
        f.write('MODEL OUTPUT FILE\n')
        f.close()
        df.to_csv(output_file, encoding='ascii', mode='a')


if __name__ == "__main__":
    IOWrapper()

