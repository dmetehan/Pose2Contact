import json
import os
import random
import numpy as np
import yaml
import argparse
from time import sleep
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from main import update_parameters, init_parser
from src.processor import Processor
import matplotlib.pyplot as plt


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
    decided_loss_weights = [{'12': 0, '6x6': 0, '42': 0, '21x21': 1},
                            {'12': 0, '6x6': 0.5, '42': 0, '21x21': 0.5},
                            {'12': 0, '6x6': 0, '42': 0.5, '21x21': 0.5},
                            {'12': 0.25, '6x6': 0.25, '42': 0.25, '21x21': 0.25}]
    # focusing on 6x6
    # decided_loss_weights = [{'12': 0, '6x6': 1, '42': 0, '21x21': 0},
    #                         {'12': 0.5, '6x6': 0.5, '42': 0, '21x21': 0}]
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


def read_results(out_file):
    results = []
    with open(out_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results):
    ordered_keys = ['6+6', '6x6', '21+21', '21x21']
    ordered_keys_2 = ['12', '6x6', '42', '21x21']
    for key in ordered_keys:
        print(f"{key:^6s}", end='&\t')
    for key in ordered_keys:
        print(f"{key:^14s}", end='&\t')
    print()
    for res in results:
        params = res[-1]
        combined = {key: [res[0][key]] for key in res[0] if key != "best_epoch"}
        for one_result in res[1:-1]:
            for key in combined:
                combined[key].append(one_result[key])
        # print(params)
        for key in ordered_keys_2:
            print(f"{f'{params[key]}':^6s}", end='&\t')
        for key in ordered_keys_2:
            key = f'jaccard{key}'
            print(f"{f'{np.mean(combined[key])*100:.2f}':>6s} ({np.std(combined[key])*100:.2f})", end=' &\t')
        print()


def main():
    out_file = 'cross_val_results.txt'
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default
    # Waiting to run
    sleep(args.delay_hours * 3600)
    if args.analyze_results:
        results = read_results(out_file)
        analyze_results(results)
    else:
        paramset = set_hyperparameter_generators()
        for key in paramset:
            for model_args in paramset[key]():
                args.model_args[key] = model_args
                best_states = run_cross_validation(args)
                with open(out_file, 'a+') as f:
                    f.write(json.dumps(best_states + [args.model_args[key]]))
                    f.write('\n')


if __name__ == '__main__':
    main()
