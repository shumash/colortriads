#!/usr/bin/env python

import sys
import os
import math
import numpy as np
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Select one best model for each image in a result set.')

    # DATA -----------------------------------------------------------------------
    parser.add_argument(
        '--base_dir', action='store', type=str, required=True,
        help='Base experiment directory')
    parser.add_argument(
        '--models', action='store', type=str, required=True,
        help='CSV list of experiment subdirectories')
    parser.add_argument(
        '--resultset', action='store', type=str, required=True,
        help='Result set prefix (already evaluated on).')
    parser.add_argument(
        '--output', action='store', type=str, required=True,
        help='File to write one best model to')

    args = parser.parse_args()

    models = args.models.strip().split(',')

    file_scores = {}
    best_model = {}

    for model in models:
        scores_file = os.path.join(args.base_dir, model, args.resultset, 'file_losses.txt')
        if not os.path.isfile(scores_file):
            raise RuntimeError('File DNE: %s' % scores_file)
        nalphas = int(model[1])
        print ('Model %s had %d alphas' % (model, nalphas))

        with open(scores_file) as f:
            first = True
            for line in f:
                if first:
                    first = False
                else:
                    parts = [x for x in line.strip().split() if len(x) > 0]
                    if len(parts) > 0:
                        fname = parts[1]
                        loss = float(parts[-1])

                        if fname not in file_scores:
                            file_scores[fname] = {}

                        file_scores[fname][model] = (loss, nalphas)

    ofile = open(args.output, 'w')
    for (fname,scores) in file_scores.iteritems():
        if len(scores) != len(models):
            print(scores)
            raise RuntimeError('Expected %d scores for %f, got %d' % (len(models), fname, len(scores)))

        best_model = None
        best_score = -1
        for (model,loss) in scores.iteritems():
            score = loss[0] + 100 * loss[1]
            if (best_model is None) or (score < best_score):
                best_model = model
                best_score = score

        print('Best %s %s %0.3f' % (fname, best_model, best_score))
        ofile.write('%s %s %0.3f\n' % (fname, best_model, best_score))

    ofile.close()
