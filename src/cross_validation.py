import json
import os
import random

import yaml
import argparse
from time import sleep
import sys
if os.path.abspath(os.path.curdir)[-3:] == 'src':
    sys.path.append(os.path.abspath('..'))
from main import update_parameters, init_parser
from src.processor import Processor

LOSS_WEIGHTS = {'42': 0.15, '12': 0.05, '21x21': 0.7, '6x6': 0.1}


def random_loss_weights_generator():
    # this function generates 4 values between 0-1 and they sum up to 1
    separators = [random.randint(0, 10) for _ in range(3)]  # this divides 0-1 region into 0.1 steps with 3 separators
    separators = sorted(separators)
    yield {'42': 0.1 * separators[0],
           '12': 0.1 * (separators[1] - separators[0]),
           '21x21': 0.1 * (separators[2] - separators[1]),
           '6x6': 0.1 * (10 - separators[2])}


def run_cross_validation(args):
    print(args)
    best_states = []
    for f in range(1, 5):
        args.dataset_args['fold'] = f'fold{f}'
        p = Processor(args)
        best_states.append(p.start())
    return best_states


def set_hyperparameter_generators():
    paramset = {}
    paramset['loss_weights'] = random_loss_weights_generator
    return paramset


def main():
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default
    # Waiting to run
    sleep(args.delay_hours * 3600)
    outfile = open('cross_val_results.txt', 'a+')
    paramset = set_hyperparameter_generators()
    for key in paramset:
        for _ in range(25):
            args.model_args[key] = next(paramset[key]())
            best_states = run_cross_validation(args)
            outfile.write(json.dumps(best_states))
            outfile.write('\n')
    outfile.close()


if __name__ == '__main__':
    main()
