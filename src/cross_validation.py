import json
import os
import random

import yaml
import argparse
from time import sleep
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from main import update_parameters, init_parser
from src.processor import Processor


def random_loss_weights_generator():
    for _ in range(100):
        # this function generates 4 values between 0-1 and they sum up to 1
        separators = [random.randint(0, 10) for _ in range(3)]  # this divides 0-1 region into 0.1 steps with 3 separators
        separators = sorted(separators)
        yield {'42': 0.1 * separators[0],
               '12': 0.1 * (separators[1] - separators[0]),
               '21x21': 0.1 * (separators[2] - separators[1]),
               '6x6': 0.1 * (10 - separators[2])}


def decided_weights_generator():
    # Loss weights decided during the meeting!
    decided_loss_weights = [{'42': 0.25, '12': 0.25, '21x21': 0.25, '6x6': 0.25},
                            {'42': 0.35, '12': 0.15, '21x21': 0.35, '6x6': 0.15},
                            {'42': 0.45, '12': 0.05, '21x21': 0.45, '6x6': 0.05},
                            {'42': 0.15, '12': 0.15, '21x21': 0.35, '6x6': 0.35},
                            {'42': 0.05, '12': 0.05, '21x21': 0.45, '6x6': 0.45},
                            {'42': 0, '12': 0, '21x21': 1, '6x6': 0},
                            {'42': 0.5, '12': 0, '21x21': 0.5, '6x6': 0},
                            {'42': 0, '12': 0.5, '21x21': 0.5, '6x6': 0},
                            {'42': 0, '12': 0, '21x21': 0.5, '6x6': 0.5}]
    for loss_weights in decided_loss_weights:
        yield loss_weights


def run_cross_validation(args):
    print(args)
    best_states = []
    for f in range(1, 5):
        args.dataset_args['fold'] = f'fold{f}'
        # args.scheduler_args['cosine']['max_epoch'] = 6  # debugging purposes
        p = Processor(args)
        best_states.append(p.start())
    return best_states


def set_hyperparameter_generators():
    paramset = {}
    # paramset['loss_weights'] = random_loss_weights_generator
    paramset['loss_weights'] = decided_weights_generator
    return paramset


def main():
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default
    # Waiting to run
    sleep(args.delay_hours * 3600)
    paramset = set_hyperparameter_generators()
    for key in paramset:
        for model_args in paramset[key]():
            args.model_args[key] = model_args
            best_states = run_cross_validation(args)
            with open('cross_val_results.txt', 'a+') as f:
                f.write(json.dumps(best_states + [args.model_args[key]]))
                f.write('\n')


if __name__ == '__main__':
    main()
