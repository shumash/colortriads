#!/usr/bin/env python

import random
import argparse
import os
import math
from skimage.io import imread,imsave

import util.hist as hist

from learn.tf.color.data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress patches into a few files')
    parser.add_argument(
        '--filelist', action='store', type=str, required=True,
        help='File containing lines with location of data files; or a csv list of files.')
    parser.add_argument(
        '--data_dir', action='store', type=str, required=True,
        help='Directory, relative to which the input files are.')
    parser.add_argument(
        '--img_width', action='store', type=int, required=True,
        help='When input is an image, scales and crops image to this size.')
    parser.add_argument(
        '--patch_width', action='store', type=int, required=True,
        help='If > 0, will compute per-patch histograms and then take a max.')
    parser.add_argument(
        '--patch_range', action='store', type=str, default=None,
        help='To randomize size, use min,max.')
    parser.add_argument(
        '--image_output_dir', action='store', type=str, default=None)
    parser.add_argument(
        '--output_filelist', action='store', type=str, default=None)
    parser.add_argument(
        '--generate_patches', action='store', type=bool, default=False)
    parser.add_argument(
        '--patches_per_image', action='store', type=int, default=10)
    parser.add_argument(
        '--compute_entropy', action='store', type=bool, default=False)
    parser.add_argument(
        '--center_bias', action='store', type=bool, default=False)
    args = parser.parse_args()

    ofile = None
    if args.output_filelist is not None:
        ofile = open(args.output_filelist, 'w')

    # Read all the patches
    if args.generate_patches:
        if args.patch_range is not None:
            patch_range = [int(x) for x in args.patch_range.split(',')]
            assert len(patch_range) == 2, "Failed to parse proper path range from %s" % args.patch_range
        else:
            patch_range = None

        helper = RandomPatchDataHelper(args.filelist, args.data_dir, 0,
                                       img_width=args.img_width, patch_width=args.patch_width,
                                       patch_range=patch_range, center_bias=args.center_bias)
        for f in range(len(helper.filenames)):
            for i in range(args.patches_per_image):
                start_col, start_row, rwidth, rwidth, patch = helper.random_patch_for_idx(f)

                entropy = ''
                if args.compute_entropy:
                    idx,counts = hist.compute_3d_histogram(patch, 10)
                    entropy = ' %0.4f' % hist.compute_histogram_entropy(counts.astype(np.float32), eps=0)

                if ofile:
                    ofile.write('%s %d,%d,%d,%d%s\n' % (helper.orig_filenames[f],
                                                        start_col, start_row, rwidth, rwidth, entropy))

                if args.image_output_dir is not None:
                    fname = os.path.join(args.image_output_dir, 'im_%s_i%04d_p%03d.png' % (entropy.strip(), f, i))
                    imsave(fname, patch)

    else:
        helper = SpecificPatchDataHelper(args.filelist, args.data_dir, 0,
                                         img_width=args.img_width, patch_width=args.patch_width)

        # [Npatches x patch_width x patch_height x 3]
        all_patches = helper.get_all()
        if args.image_output_dir is not None:
            for i in range(all_patches.shape[0]):
                fname = os.path.join(args.image_output_dir, 'im_%04d.png' % i)
                imsave(fname, all_patches[i, :, :, :])



