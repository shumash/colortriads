#!/usr/bin/env python

import sys
import os
import numpy as np
import util.io
import argparse
import util.hist as hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Som On Color Samples')
    parser.add_argument(
        '--nbins', action='store', type=int, default=10)
    parser.add_argument(
        '--img', action='store', type=str, required=True)
    parser.add_argument(
        '--output', action='store', type=str, required=True)
    args = parser.parse_args()

    img = util.io.read_float_img(args.img)
    (idx, counts) = hist.compute_3d_histogram(img, args.nbins)
    hist.write_3d_histogram(idx, counts, args.nbins, args.output)
