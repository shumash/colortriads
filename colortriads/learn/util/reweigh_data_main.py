#!/usr/bin/env python

import random
import argparse
import os
import math
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split training set')
    parser.add_argument(
        '--frequency_fractions', action='store', type=str, required=True,
        help='For each source, a relative frequency component, e.g. 1.0,1.3,4.5.')
    parser.add_argument(
        '--input_files', action='store', type=str, required=True,
        help='CSV list of input files with one datapoint per line')
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True)
    args = parser.parse_args()

    fracs = np.array([float(x) for x in args.frequency_fractions.strip().split(',') ], np.float32)
    fracs = fracs / np.min(fracs)  # all fractions >= 1.0
    print('Using fractions %s' % str(fracs))

    # Read all the input files
    filenames = args.input_files.strip().split(',')
    assert len(filenames) == len(fracs), 'Must supply equal number of sources and fractions'
    for i in range(len(filenames)):
        fname = filenames[i]
        weight = fracs[i]
        oname = 'reweighted_%s' % os.path.basename(fname)
        print('Reweighing %s with %0.3f' % (oname, weight))

        with open(fname) as f:
            new_lines = [l.strip() for l in f.readlines()]
            if len(new_lines) == 0:
                raise RuntimeError('Read nothing from %s' % fname)

            # Now we sample from the poisson
            with open(os.path.join(args.output_dir, oname), 'w') as ofile:
                for line in new_lines:
                    num_samples = int(1 + np.random.poisson(weight - 1.0))  # we ensure a count of at least 1
                    for s in range(num_samples):
                        ofile.write(line + '\n')

