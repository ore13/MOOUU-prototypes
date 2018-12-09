import os
import sys
import argparse
import numpy as np
import pandas as pd
import Test_suite as ts

test_dict = {"zdt1": ts.ZDT1, "zdt2": ts.ZDT2, "zdt3": ts.ZDT3, "zdt4": ts.ZDT4, "zdt6": ts.ZDT6}


def parse():
    """ parse the command args

    Parameters
    ----------
        None

    Returns
        args : ArgumentParser instance parsed arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_function", nargs=1, dest="test_function", default="zdt1")
    parser.add_argument("--input", nargs=1, default="input.dat", dest="input_file")
    parser.add_argument("--output", nargs=1, default="output.dat", dest="output_file")

    return parser.parse_args()


def run_through_wrapper(args):
    """run a given ZDT benchmark via an input and output file

    Parameters
    ----------
        args : ArgumentParser instance parsed arguments from the command line

    Returns
    -------
        None
    """

    test_function = args.test_function.lower()
    if test_function not in test_dict.keys():
        raise Exception("'test_function' {0} not found in test suite", format(test_function))
    tf = test_dict[test_function]
    try:
        input_df = pd.read_csv(args.input_file)
    except Exception as e:
        raise Exception("error reading input file {0}:{1}".format(args.input_file, str(e)))
    if len(input_df.columns) != tf.number_decision_vars():
        raise Exception("Incorrect number of decision variables. Should be {} d. vars. got {} d. vars."
                        .format(len(input_df.columns), tf.number_decision_vars()))
    f1 = tf.f1(input_df.parval1)
    f2 = tf.f2(input_df.parval1)
    with open(args.output_file, 'w') as f:
        f.write("f1 {0:20.8E}\n".format(f1))
        f.write("f2 {0:20.8E}\n".format(f2))


def test_wrapper():
    """test that the io wrapper is doing something...

    """
    args = parse()
    names = ["par_{0:02d}".format(i) for i in range(30)]
    vals = np.zeros(30) + 0.5
    df = pd.DataFrame({"parnme": names, "parval1": vals})
    df.to_csv("input.dat")
    run_through_wrapper(args)
    assert os.path.exists(args.output_file)
    df = pd.read_csv(args.output_file, delim_whitespace=True, header=None)
    print(df)

if __name__ == "__main__":
    #run_through_wrapper()
    test_wrapper()
