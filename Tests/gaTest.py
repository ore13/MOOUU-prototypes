
import subprocess
import os

if __name__ == '__main__':
    filename = 'ga.err'
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, 'w')
    f.close()
    dir = os.getcwd()
    path = os.path.join(dir, 'GeneticAlgorithms')
    subprocess.run(['python', os.path.join(path, 'Abstract_MOO_test.py'), filename])
    subprocess.run(['python', os.path.join(path, 'NSGA_II_test.py'), filename])
    subprocess.run(['python', os.path.join(path, 'SPEA_test.py'), filename])
    subprocess.run(['python', os.path.join(path, 'SPEA_2_test.py'), filename])
    f = open(filename)
    error_occured = False
    for line in f:
        print(line)
        error_occured = True
    if not error_occured:
        print("no errors occured")
