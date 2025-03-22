#!/usr/bin/env python

import random
import argparse
import os
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split training set')
    parser.add_argument(
        '--frac_test', action='store', type=float, default=0.01)
    parser.add_argument(
        '--frac_validation', action='store', type=float, default=0.005)
    parser.add_argument(
        '--input_files', action='store', type=str, required=True,
        help='CSV list of input files with one datapoint per line')
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True)
    args = parser.parse_args()

    exact_counts = False
    if args.frac_test > 1 or args.frac_validation > 1:
        exact_counts = True
        print('Note: values > 1: treating frac_test and frac_validation as counts, not fractions')
    elif args.frac_test + args.frac_validation > 1.0:
        raise RuntimeError('Error: sum of --frac_test and --frac_validation > 1')

    # Read all the input files
    all_lines = []
    filenames = args.input_files.strip().split(',')
    for fname in filenames:
        with open(fname) as f:
            new_lines = [l.strip() for l in f.readlines()]
            if len(new_lines) == 0:
                raise RuntimeError('Read nothing from %s' % fname)
            all_lines.extend(new_lines)
    if len(all_lines) < 5:
        raise RuntimeError('Too few lines read: %s' % '\n'.join(all_lines))

    indexes = range(len(all_lines))
    random.shuffle(indexes)

    if exact_counts:
        test_num = int(args.frac_test)
        val_num = int(args.frac_validation)
    else:
        test_num = int(math.ceil(args.frac_test * len(all_lines)))
        val_num = int(math.ceil(args.frac_validation * len(all_lines)))

    with open(os.path.join(args.output_dir, 'keys_test.txt'), 'w') as f:
        for i in range(test_num):
            idx = indexes[i]
            f.write(all_lines[idx] + '\n')

    with open(os.path.join(args.output_dir, 'keys_eval.txt'), 'w') as f:
        for i in range(test_num, test_num + val_num):
            idx = indexes[i]
            f.write(all_lines[idx] + '\n')

    with open(os.path.join(args.output_dir, 'keys_train.txt'), 'w') as f:
        for i in range(test_num + val_num, len(all_lines)):
            idx = indexes[i]
            f.write(all_lines[idx] + '\n')

    print('Wrote split files to %s' % args.output_dir)
