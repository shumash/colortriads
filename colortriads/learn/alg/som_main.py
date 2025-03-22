#!/usr/bin/env python

import sys
import os
import cv2
import numpy as np
import scipy.misc
import argparse

from learn.alg.som import SOMParam, SOM, SOMPlot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Som On Color Samples')
    parser.add_argument(
        '--samples', action='store', type=str, required=True)
    parser.add_argument(
        '--out_basename', action='store', type=str, required=True)
    parser.add_argument(
        '--output_evern_N', action='store', type=int, default=-1)
    args = parser.parse_args()


    tdata = np.loadtxt(args.samples)

    som_param = SOMParam(h=32, dimension=2)
    som = SOM(tdata, som_param)
    som.trainAll()

    # SOM plotter.
    som2D_plot = SOMPlot(som2D)

    fig = plt.figure()

    # Plot 2D SOM.
    fig.add_subplot(131)
    som2D_plot.updateImage()
    plt.axis('off')
