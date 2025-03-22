#!/usr/bin/env python

import sys
import os
import numpy as np
import time
import argparse
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python import debug as tf_debug
import skimage.io
import argparse
import numpy as np
from skimage.io import imsave

import learn.tf.color.encoder as colorenc
import learn.tf.color.color_ops as color_ops
from learn.tf.log import LOG
import learn.tf.util as tf_util
import learn.tf.color.palette as palette
import learn.tf.color.encoder as encoder
import learn.tf.color.setup_args
from learn.tf.color.data import ColorGlimpsesDataHelper

import util.io
import util.img_util as img_util
import learn.tf.color.mapping as mapping


def log_tensor(t, name):
    print('{}: {} {}'.format(name, t.dtype, t.shape))

def uv_to_int(uv, nsubdiv):
    print(uv * (nsubdiv - 1))
    return np.round(uv * (nsubdiv - 1)).astype(np.int64)

def int_to_bits(input, nbits):
    bstr = format(input, 'b')
    if len(bstr) > nbits:
        raise RuntimeError('Max {} bits requested, but bin({}) returned {}'.format(nbits, input, bstr))
    bits = [ 0 for i in range(nbits - len(bstr)) ]
    bits.extend([ int(bstr[i]) for i in range(len(bstr))])
    return bits

# Can represent:
# 1 bit -> 2 numbers
# 2 bits -> 4 numbers
# 3 bits -> 8 numbers
def encode_uv(uv, nsubdiv, no_bit_efficiency=False):
    denominator = nsubdiv - 1
    uv_int = uv_to_int(uv, nsubdiv)
    log_tensor(uv_int, 'uv_int')
    print('Max int: %d' % np.max(uv_int))
    print('Nunique colors: {}'.format(np.unique(uv_int, axis=0).shape))
    #raise RuntimeError('stop')

    nbits_per_idx = int(math.ceil(math.log(nsubdiv, 2)))  # i.e. nsubdiv=8 -> must represent 8 distinct numbers
    max_bits = nbits_per_idx * 2 * uv.shape[0]
    print('Encoding with MAX BITS PER VAL %d (max total %d)' % (nbits_per_idx, max_bits))
    bit_array = np.zeros((max_bits), dtype=np.int64)
    loc = 0
    for i in range(uv.shape[0]):
        uv_val = uv_int[i, :]
        first_numerator = uv_val[0]
        max_second_numerator = denominator - first_numerator
        nbits_needed = int(math.ceil(math.log(max_second_numerator + 1, 2)))
        if no_bit_efficiency:
            nbits_needed = nbits_per_idx
        #print('Encoding %d, %d with %d bits and %d bits (for max value %d)' %
         #     (uv_val[0], uv_val[1], nbits_per_idx, nbits_needed, max_second_numerator))
        bits = int_to_bits(first_numerator, nbits_per_idx)
        if nbits_needed > 0:
            bits.extend(int_to_bits(uv_val[1], nbits_needed))
        #print('Encoded with %d bits' % len(bits))

        for j in range(len(bits)):
            bit_array[loc + j] = bits[j]
        #bit_array[loc:(loc + len(bits))] = bits
        loc = loc + len(bits)
    print('Used %d bits (max %d requred at %d bits per idx)' % (loc, nbits_per_idx * 2 * uv.shape[0], nbits_per_idx))
    encoding = np.packbits(bit_array)
    log_tensor(encoding, 'Final Encoding:')
    return encoding

def compute_losses(in_colors, recon):
    l2loss = []
    percent_loss = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', action='store', type=str, required=True,
        help='Image file or ')
    parser.add_argument(
        '--img_width', action='store', type=int, default=-1,
        help='When input is an image, scales and crops image to this size.')
    parser.add_argument(
        '--palette', action='store', type=str, required=True,
        help='File containing ascii sail')
    parser.add_argument(
        '--max_tri_subdivs', action='store', type=int, default=-1)
    parser.add_argument(
        '--output_image', action='store', type=str, default=None)
    parser.add_argument(
        '--output_encoding', action='store', type=str, default=None)
    parser.add_argument(
        '--output_binary', action='store', type=str, default=None)
    parser.add_argument(
        '--output_losses_append', action='store', type=str, default=None)
    parser.add_argument(
        '--ignore_upside_down', action='store_true', default=False)
    parser.add_argument(
        '--fake_alpha', action='store', type=str, default=None)
    parser.add_argument(
        '--decode', action='store_true')

    args = parser.parse_args()

    with open(args.palette, 'r') as f:
        pstr = f.readlines()[0]
    colors_val, wind_val, nsubdiv = palette.color_sail_from_string(pstr)
    if args.max_tri_subdivs < 0:
        args.max_tri_subdivs = nsubdiv

    # res, verts, n_tri, cweights, tri_centroids, tri_interps = \
    popts = palette.PaletteOptions(max_colors=3,
                                   max_tri_subdivs=args.max_tri_subdivs,
                                   discrete_continuous=False,
                                   use_alpha=False,
                                   wind_nchannels=3)
    popts.max_wind = 1.0

    colors = tf.placeholder(tf.float32, [1, popts.n_colors * popts.n_channels])
    nraw_ch = popts.wind_nchannels + 1 if popts.wind_nchannels > 1 else popts.wind_nchannels
    wind = tf.placeholder(tf.float32, [1, nraw_ch])

    phelper = palette.PaletteHelper(popts)
    phelper.init_deterministic_decoder(colors, wind)

    colors_val = colors_val.reshape(colors.shape)
    wind_val = wind_val.reshape(wind.shape)
    right_tri = phelper.tri_right_side_up_indicator
    log_tensor(colors_val, 'colors_val')
    log_tensor(right_tri, 'right tri')
    print(wind_val)
    print('Right side up triangles: %d' % np.sum(right_tri))

    # Next we get palette colors
    sess = tf.Session()
    vars = { 'pcolors': phelper.patch_colors }
    input_dict = { colors:colors_val, wind:wind_val }
    vals = tf_util.evaluate_var_dict(sess, vars, input_dict)
    log_tensor(vals['pcolors'], 'pcolors')
    source_colors = (vals['pcolors'].squeeze() * 255).astype(np.uint8)

    # HACK: zero out upside down triangles
    if args.ignore_upside_down:
        color_mask = np.tile(np.expand_dims(phelper.tri_right_side_up_indicator, axis=1), (1, 3))
        log_tensor(color_mask, 'color mask')
        source_colors[np.logical_not(color_mask)] = 0
        log_tensor(source_colors, 'source colors')

    if not args.decode:
        # Next find a mapping
        target = util.img_util.read_resize_square(args.input, args.img_width, dtype=np.uint8)
        log_tensor(target, 'target')
        #recon = mapping.make_approx_mapping(source_colors, target)
        recon, ids_map = mapping.make_exact_mapping(source_colors, target, return_idx=True, use_lab=True)
        log_tensor(recon, 'recon')
        log_tensor(ids_map, 'ids_map')
        if args.output_image is not None:
            imsave(args.output_image, recon)

        imsave('/tmp/ids.png', (ids_map * 255 / source_colors.shape[0]).astype(np.uint8))
        uv = phelper.color_idx_to_color_bary(ids_map.reshape(-1))
        if args.output_binary is not None:
            uv_bin = phelper.color_idx_to_center_bary(ids_map.reshape(-1))

            alphas=None
            if args.fake_alpha is not None:
                alphas = [img_util.read_resize_square(args.fake_alpha, target.shape[0], nchannels=4)]
                if len(alphas[0].shape) > 2:
                    alphas[0] = alphas[0][:, :, 3]
                imsave('/tmp/fake_alpha.png', alphas[0])
                log_tensor(alphas[0], 'fake alpha')

            encoder.write_binary_color_sail_rig_file(
                args.output_binary, target,
                [encoder.make_palette_dict_binary_output(colors_val, args.max_tri_subdivs, wind_val)],
                 [uv_bin], alphas=alphas)

        log_tensor(uv, 'uv')
        uv_img = np.concatenate([np.reshape(uv, [target.shape[0], target.shape[1], 2]),
                                 np.zeros([target.shape[0], target.shape[1], 1])], axis=2)
        imsave('/tmp/uv.png', (uv_img * 255).astype(np.uint8))

        encoding = encode_uv(uv, nsubdiv, no_bit_efficiency=True)

        if args.output_encoding is not None:
            with open(args.output_encoding, 'wb') as f:
                f.write(encoding.tobytes())
            print('Wrote: %s' % args.output_encoding)
            if encoding.nbytes == target.shape[0] * target.shape[1]:
                imsave(args.output_encoding + '.png', encoding.reshape(target.shape[0:2]))
                print('Also wrote as image: %s' % (args.output_encoding + '.png'))
    else:
        raise RuntimeError('Decode not implemented')
