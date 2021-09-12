import os
from configparser import ConfigParser
import argparse 
import copy

# takes in filepath to default .ini
arg_parser = argparse.ArgumentParser(description='Process filepath and files.')
arg_parser.add_argument('default_ini', help = "filpath leading to default initializations", default='default.ini')
arg_parser.add_argument('testfile', help = 'file with testing options', default='test.ini')
defaultfile = arg_parser.parse_args().default_ini # default file
testfile = arg_parser.parse_args().testfile # save destination for metrics

# Parses the tests we want (assumes testfile is space delimited)
test_parser = ConfigParser()
default_parser = ConfigParser()

test_parser.read(testfile)
default_parser.read(defaultfile)

# reads the test parser for values we will need to change
# assumes testfile is space delimited between values
# if multiple values are to be changed, they should have the same length


metrics_root = 'metrics'
if not os.path.exists(metrics_root):
    os.mkdir(metrics_root)
# an inefficient algorithm
for test_type in test_parser.sections():
    # get all the changes we want
    all_changes = test_parser.items(test_type)
    num_params_to_change = len(all_changes)
    params = [tup[0] for tup in all_changes]
    values = [tup[1].split(' ') for tup in all_changes]
    print(f"Test Type {test_type}:")
    for test_num in range(len(values[0])): # iterate n number of tests
        # write temp_parser to the appropriate 
        test_path = os.path.join(metrics_root, f'{test_type}_tests' )
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        # make a deep copy of default which we will change and write
        print(f"\ttest {test_num}:")
        temp_parser = copy.deepcopy(default_parser)
        for i in range(num_params_to_change): # iterate through our parameters to change
            for sections in temp_parser.sections(): # have to iterate to figure out where the update is
                if temp_parser.has_option(sections, params[i]): # if the option exists, overwrite it
                    temp_parser.set(sections, params[i], values[i][test_num ])
                    print(f"\t\treplaced {params[i]}: {values[i][test_num]}")
                    
        individual_test_path = os.path.join(test_path, f'test{test_num}')
            
        if not os.path.exists(individual_test_path):
            os.mkdir(individual_test_path)

        #with open
        with open(os.path.join(individual_test_path, 'params.ini'), 'w') as f:
            temp_parser.write(f)
        






