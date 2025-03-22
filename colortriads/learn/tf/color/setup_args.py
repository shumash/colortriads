import sys
import os
import numpy as np
import random
import argparse
import tensorflow as tf
import scipy.misc
import math

import learn.tf.color.encoder as colorenc
import learn.tf.color.discriminator as colordisc
from tensorflow.python import debug as tf_debug
import learn.tf.color.color_ops as color_ops
from learn.tf.log import LOG
import learn.tf.util as tf_util
import learn.tf.color.palette as palette
import learn.tf.color.setup_args
from learn.tf.color.data import HistDataHelper,ImageDataHelper,ComputedHistDataHelper,ExternalPaletteDataHelper
from learn.tf.color.data import RandomPatchDataHelper,RandomGrayscaler,SpecificPatchDataHelper


INPUT_MODES = { 0: 'read pre-computed histogram from file',
                1: 'read and scale image and compute histogram to use in tf',
                2: 'read and scale image and input directly to tf',
                3: 'generate synthetic palette image using external colors ' +
                '(our rep with random level up for each triangle to max_tri_subdivs)',
                4: 'get a random patch from input images' }

def add_graph_flags(parser):
    parser.add_argument(
        '--restore_palette_graph', action='store', type=str, default='',
        help='If set, will restore FC palette graph variables from a checkpoint.')

            # INPUT OPTIONS --------------------------------------------------------------
    parser.add_argument(
        '--input_mode', action='store', type=int, required=True,
        help='Input modes: ' + str(INPUT_MODES))

        # ENCODER OPTIONS ------------------------------------------------------------
    parser.add_argument(
        '--encoder_mode', action='store', type=int, default=True,
        help='Determines type of graph to use: ' +
        '0 - encode histogram with FC layers to produce palette,' +
        '1 - encode image with Conv and FC layers (only for input_mode=2,3) to produce palette,' +
        '2 - img 2 img encoder with vanilla UNET to select regions and a palette per each,' +
        '3 - vgg-to-unet architecture to select regions and palette per each, ' +
        '4 - encode histogram with conv layers.')
    parser.add_argument(
        '--palette_graph_mode', action='store', type=int, default=True,
        help='Accepts one of --encoder_mode its relevant to palette encoding (0,4)')

    parser.add_argument(
        '--rnn', action='store', type=bool, default=False,
        help='For encoder modes 2, 3 uses rnn architecture.')
    parser.add_argument(
        '--n_masks', action='store', type=int, default=4,
        help='If graph is set up to output masks, number of masks to output.')

    # Releavant to modes 1, 2
    parser.add_argument(
        '--img_width', action='store', type=int, default=300,
        help='When input is an image, scales and crops image to this size.')
    # Relevant to mode 2
    parser.add_argument(
        '--patch_width', action='store', type=int, default=-1,
        help='If > 0, will compute per-patch histograms and then take a max.')
    # Relevant to modes 0,1,2
    parser.add_argument(
        '--n_hist_bins', action='store', type=int, default=10,
        help='Number of histogram bins to use.')

    # Relevant to all encoder modes
    # parser.add_argument(
    #     '--fc_sizes', action='store', type=str, required=True,
    #     help='Fully connected layer sizes, exluding the last encoding size, as CSV ints, ' +
    #     'e.g. 700,200,50.')

    parser.add_argument(
        '--palette_layer_specs', action='store', type=str, required=True,
        help='A ; separated list of layer opts for palette, e.g. ' +
        'in the case of FC (palette_graph_mode=0) this could be 700;200;50, ' +
        'and in the case of Conv (palette_graph_mode=4) this could be ' +
        '3,64,2;3,128,2;3,256,3;1,64,1')

    # Relevant to encoder mode 1
    parser.add_argument(
        '--conv_filter_sizes', action='store', type=str, default='4,4,4',
        help='Filter sizes for the convolutional layers as CSV ints (relevant to --encoder_mode=1)')
    parser.add_argument(
        '--conv_filter_counts', action='store', type=str, default='64,128,256',
        help='Number of filters for each convolutiona layer as CSV ints (relevant to --encoder_mode=1)')

        # PALETTE OPTIONS -------------------------------------------------------------
    parser.add_argument(
        '--max_colors', action='store', type=int, default=3,
        help='Maximum number of platte color vertices.')
    parser.add_argument(
        '--max_tri_subdivs', action='store', type=int, default=2,
        help='Maximum number of palette triangle subdivisions.')
    parser.add_argument(
        '--encode_subdiv_levels', action='store', type=bool, default=False,
        help='If on, sets up the graph to output subdivision level for each palette triangle.')
    parser.add_argument(
        '--encode_alpha', action='store', type=bool, default=False,
        help='If on, encodes colors with alpha, effectively selecting which colors to turn on.')
    parser.add_argument(
        '--wind_num_channels', action='store', type=int, default=0,
        help='Control for wind usage.')

    #AMLAN ADDITIONS
    parser.add_argument(
        '--mask_softmax_temp', action='store', type=float, default=1.0,
        help='Control for temperature parameter in the mask softmax.')
    parser.add_argument(
        '--mask_dropout', action='store', type=float, default=0.0,
        help='Control for dropout in the mask input graph.')


def setup_encoder_graph(args):
    def csv_to_elem(csv_str):
        elems = [int(x) for x in csv_str.strip().split(',')]
        if len(elems) == 1:
            return elems[0]
        else:
            return tuple(elems)

    #fc_sizes
    palette_layer_specs = [csv_to_elem(x) for x in args.palette_layer_specs.strip().strip("'").split(';')]
    conv_info = zip([int(x) for x in args.conv_filter_sizes.strip().split(',')],
                    [int(x) for x in args.conv_filter_counts.strip().split(',')])

    vae = colorenc.ColorAutoencoder(args.n_hist_bins)
    popts = palette.PaletteOptions(max_colors=args.max_colors,
                                   max_tri_subdivs=args.max_tri_subdivs,
                                   discrete_continuous=args.encode_subdiv_levels,
                                   use_alpha=args.encode_alpha,
                                   wind_nchannels=args.wind_num_channels)

    # Initialize input ---------------------------------------------------------
    if args.input_mode in [0, 1]:
        vae.init_hist_input()
    elif args.input_mode == 2:
        vae.init_image_input(img_width=args.img_width)
        vae.add_histogram_computation(patch_width=args.patch_width)
    elif args.input_mode == 3:
        vae.init_image_input(img_width=args.img_width)
        vae.add_histogram_computation(patch_width=args.patch_width, zero_out_white=True)
    elif args.input_mode == 4:
        vae.init_image_input(img_width=args.patch_width)
        vae.add_histogram_computation()
    else:
        raise RuntimeError('Unknown --input_mode %s' % str(args.input_mode))


    # Initialize graph ---------------------------------------------------------
    modifier = setup_input_modifier(args)
    
    # Set dropout to None to use 3 channle mask graph
    if args.mask_dropout < 0:
        args.mask_dropout = None

    if args.encoder_mode == 0:
        vae.init_hist_encoder_graph(palette_opts=popts, fcsizes=palette_layer_specs)
    elif args.encoder_mode == 1:
        vae.init_image_encoder_graph(palette_opts=popts, fcsizes=palette_layer_specs, conv_layers=conv_info)
    elif args.encoder_mode == 2:
        vae.init_vanilla_masks_graph(
            modifier.nchannels, palette_graph_arch=args.palette_graph_mode, palette_opts=popts,
            npalettes=args.n_masks, use_rnn=args.rnn, palette_layer_specs=palette_layer_specs,
            mask_softmax_temp=args.mask_softmax_temp, dropout=args.mask_dropout)
    elif args.encoder_mode == 3:
        vae.init_vgg_masks_graph(
            modifier.nchannels, palette_graph_arch=args.palette_graph_mode, palette_opts=popts,
            npalettes=args.n_masks, use_rnn=args.rnn, palette_layer_specs=palette_layer_specs,
            mask_softmax_temp=args.mask_softmax_temp, dropout=args.mask_dropout)
    elif args.encoder_mode == 4:
        vae.init_3d_hist_encoder_graph(palette_opts=popts, layer_specs=palette_layer_specs)
    else:
        raise RuntimeError('Unknown --encoder_mode %s' % str(args.encoder_mode))

    return vae, popts


def add_discriminator_flags(parser):
    # No flags yet, but will soon appear
    pass


def setup_discriminator(args, reuse=False):
    disc = colordisc.Discriminator()
    return disc


def add_session_flags(parser):
    parser.add_argument(
        '--run_dir', action='store', type=str, required=True,
        help='Directory where to put the model checkpoints, etc.')
    parser.add_argument(
        '--overwrite', action='store', type=bool, default=False,
        help='Set to true to overwrite run_dir (else will restore the net weights)')
    parser.add_argument(
        '--debug_sess', action='store', type=bool, default=False)
    parser.add_argument(
        '--debug_level', action='store', type=bool, default=1,
        help='Accepts the level of debugging: ' +
             '0 (no debugging), ' +
             '1 (minor diagnostics, more frequent model saving), ' +
             '2 (major diagnostics -- very slow -- and frequent saving)')


def setup_session_and_helper(args, log_vars):
    sess = tf.Session()
    if args.debug_sess:
        LOG.info('Enabling debugging session')
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    if args.debug_level == 0:
        save_frequency = 1000
        summary_frequency = 1000
    elif args.debug_level == 1:
        save_frequency = 1000
        summary_frequency = 200
    else:
        save_frequency = 100
        summary_frequency = 100

    helper = tf_util.TrainHelper(
        args.run_dir, sess,
        save_summary_every_n=summary_frequency,
        save_model_every_n=save_frequency,
        overwrite=args.overwrite,
        log_vars=log_vars,
        log_every_n=(summary_frequency if args.debug_level <= 1 else 2))

    return sess, helper


def add_testdata_flags(parser):
    parser.add_argument(
        '--testdata', action='store', type=str, required=True,
        help='File containing lines with location of data files; or a csv list of files.')
    parser.add_argument(
        '--max_test_batch_size', action='store', type=int, default=50,
        help='Sets the maximum number of test values to process concurrently; if testdata ' +
        'is bigger, will break it up')


def read_testdata(args):
    test_data = []  # Tuples ("testname" : [ batch0_input], [batch1_input] ...)
    test_files = args.testdata.strip().split(',')
    for f in test_files:
        data_name = os.path.basename(f).split('.')[0]
        data_reader = learn.tf.color.setup_args.setup_data_source(f, args, is_test=True)
        tbatches = []
        n_tbatches = int(math.ceil(len(data_reader.filenames) / (0.0 + args.max_test_batch_size)))
        tbatch_sizes = []
        for b in range(n_tbatches):
            start_b = b * args.max_test_batch_size
            end_b = min(len(data_reader.filenames), start_b + args.max_test_batch_size)
            data_in_batch = data_reader.get_specific_batch(range(start_b, end_b))
            tbatch_sizes.append(data_in_batch.shape[0])
            tbatches.append(data_in_batch)
        test_data.append((data_name, tbatches))
        LOG.info(
            'TESTSETLOG Read "%s" test data %d of %d batches: %s' %
            (data_name, len(data_reader.filenames), len(tbatches), ' '.join([str(x) for x in tbatch_sizes])))
    LOG.info('Processed %d test data sources' % len(test_files))

    return test_data


def add_loss_flags(parser, required=True):
    # LOSS OPTIONS --------------------------------------------------------------
    parser.add_argument(
        '--losses', action='store', type=str, required=required,
        help='CSV list of the following losses (as name): ' +
        'KL_SYM - symmetrized histogram KL divergence, ' +
        'KL_PAL - kl divergence (palette || image), ' +
        'L2RGB - L2 reconstruction loss in RGB, ' +
        'L2LAB - L2 reconstruction loss in LAB, ' +
        'RECON_PERCENT - Epercent loss; not allowed for training, ' +
        'SOFT_GOODNESS - squashed L1 lab error, ' +
        'HARD_GOODNESS - more squashed L1 lab error, ' +
        'REG - regularization on all trainable weights (L2), ' +
        'ALPHA_TV - total variation regularization on overall alpha (if applicable), ' +
        'ALPHA_NEG_MAX - negative sum of maximal alpha values in each channel, ' +
        'ALPHA_BIN_SQUASH - squasher on the alpha, ' +
        'ALPHA_BINARY_V2 - another squasher on the alpha, ' +
        'ALPHA_ENTROPY - entropy, ' +
        'ALPHA_MAXMINDIFF - self explanatory loss on alpha.')
    parser.add_argument(
        '--loss_weights', action='store', type=str, required=required,
        help='CSV list of weight for the losses.')
    parser.add_argument(
        '--blend_l2_alpha_loss', action='store', type=bool, default=True,
        help="If true, uses alpha weights to composite colors; else uses maximal alpha's colors.")

    # COST OPTIONS -------------------------------------------------------------
    # parser.add_argument(
    #     '--area_weighted_hist', action='store', type=bool, default=False,
    #     help='If true, and the graph outputs subdivision levels, weighs palette hist by patch areas.')


def setup_loss(vae, args, test_summary_collections=['test']):
    if args.losses is None:
        LOG.info('No losses specified')
        return None

    # Initialize loss --------------------------------------------------------------------------------------------------
    losses = args.losses.split(',')
    loss_weights = [ float(x) for x in args.loss_weights.split(',') ]
    if len(losses) != len(loss_weights):
        raise RuntimeError(
            'Losses and loss weights have different lengths: %s vs. %s' %
            (args.losses, args.loss_weights))

    loss = 0.0
    losses_i = []  # Save so that we can create a single scope, arg!
    for i in range(len(losses)):
        if losses[i] == 'KL_SYM':
            loss_i = vae.init_kl_loss()
        elif losses[i] == 'KL_PAL':
            loss_i = vae.init_onesided_kl_loss()
        elif losses[i] == 'L2RGB':
            loss_i = vae.init_l2rgb_loss(blend=args.blend_l2_alpha_loss)
        elif losses[i] == 'L2LAB':
            loss_i = vae.init_l2lab_loss()
        elif losses[i] == 'SOFT_GOODNESS':
            loss_i = vae.init_goodness_loss(is_soft=True)
        elif losses[i] == 'HARD_GOODNESS':
            loss_i = vae.init_goodness_loss(is_soft=False)
        elif losses[i] == 'REG':
            loss_i = tf_util.init_l2_reg()
        elif losses[i] == 'ALPHA_TV':
            loss_i = vae.init_alpha_tv_loss()
        elif losses[i] == 'ALPHA_BIN_SQUASH':
            loss_i = vae.init_alpha_binary_loss()
        elif losses[i] == 'ALPHA_NEG_MAX':
            loss_i = vae.init_alpha_neg_max_loss()
        elif losses[i] == 'ALPHA_BINARY_V2':
            loss_i = vae.init_alpha_binary_v2_loss()
        elif losses[i] == 'ALPHA_ENTROPY':
            loss_i = vae.init_alpha_entropy_loss()
        elif losses[i] == 'ALPHA_MAXMINDIFF':
            loss_i = vae.init_alpha_maxmindiff_loss()
        elif losses[i] == 'RECON_PERCENT':
            loss_i = vae.init_percent_reconstruction_loss()
            if loss_weights[i] > 0:
                raise RuntimeError('Cannot use RECON_PERCENT loss for training; set its weight to 0 to track only')
        else:
            raise RuntimeError('Unknown loss type: %s' % losses[i])
        loss_i = tf.identity(loss_i, losses[i])
        if loss_weights[i] > 0:
            LOG.info('Using loss %s' % losses[i])
            loss += loss_weights[i] * loss_i
        else:
            LOG.warning('Loss %s has zero weight -- only using for tracking' % losses[i])
        losses_i.append(tf.identity(loss_i, name=losses[i]))

    loss = tf.identity(loss, name='total_loss')

    collections = [x for x in test_summary_collections]
    collections.append('train')
    for n in collections:
        with tf.name_scope('%s_loss' % n):
            tf.summary.scalar('total_loss', loss)
            for i in range(len(losses_i)):
                tf.summary.scalar(losses[i], losses_i[i])

    return loss, losses_i


def add_data_config_flags(parser):
    parser.add_argument(
        '--data_input_field', action='store', type=int, default=0,
        help='Which field of the input files corresponds to the relevant file.')
    parser.add_argument(
        '--data_dir', action='store', type=str, required=True,
        help='Directory, relative to which the input files are.')
    parser.add_argument(
        '--cache_dir', action='store', type=str, default='',
        help='Global cache directory; enables caching of files (e.g. resized images) for'
             'some --input_mode values')


def setup_data_source(datafile, args, is_test=False):
    if args.input_mode == 0:
        res = HistDataHelper(datafile, args.data_dir, args.data_input_field)
    elif args.input_mode == 1:
        hist_field = 1 if args.data_input_field == 0 else 0
        res = ComputedHistDataHelper(
            datafile, args.data_dir, args.data_input_field,
            nbins=args.n_hist_bins, img_width=args.img_width, hist_field_num=hist_field,
            hist_save_dir='/tmp/hists')
    elif args.input_mode == 2:
        res = ImageDataHelper(datafile, args.data_dir, args.data_input_field,
                              img_width=args.img_width)
    elif args.input_mode == 4:
        if is_test:
            res = SpecificPatchDataHelper(datafile, args.data_dir, args.data_input_field,
                                          img_width=args.img_width, patch_width=args.patch_width)
        else:
            res = RandomPatchDataHelper(datafile, args.data_dir, args.data_input_field,
                                        img_width=args.img_width, patch_width=args.patch_width,
                                        patch_range=(args.patch_width/2,args.img_width/2), center_bias=True)
    elif args.input_mode == 3:
        res = ExternalPaletteDataHelper(datafile, img_width=args.img_width,
                                        max_tri_subdivs=args.max_tri_subdivs)
    else:
        raise RuntimeError('Unknown --input_mode %s' % str(args.input_mode))

    if len(args.cache_dir) != 0:
        if args.input_mode == 2 or args.input_mode == 3:
            LOG.info('Enabling data caching')
            res.enable_file_cache(args.cache_dir)

    return res


def single_image_data_source(imgfile, img_width, data_dir=''):
    res = ImageDataHelper(datafile=None, data_dir=data_dir, field_num=0, img_width=img_width)
    try:
        res.add_single_abspath(imgfile, do_checks=True)
    except RuntimeError as e:
        LOG.error('Failed to get a data source: ' + e.message)
        return None
    return res


def setup_data_sources(test_files, args, is_test):
    res = {}
    for f in test_files:
        data_name = os.path.basename(f).split('.')[0]
        data_reader = setup_data_source(f, args, is_test=is_test)
        res[data_name] = data_reader
    return res


def setup_input_modifier(args):
    try:
        if args.input_mode == 2:
            return RandomGrayscaler(
                args.img_width, random_bw_fraction=args.frac_bw_input, color_glimpses=args.add_color_glimpses)
        else:
            return RandomGrayscaler(args.img_width, -1, -1)  # noop
    except:
        return RandomGrayscaler(args.img_width, -1, -1)



def create_argument_string(args):
    s = ' \\\n'.join(['--%s=%s' % (x[0], str(x[1]).replace('False', '')) for x in args.__dict__.items()]) + '\n'
    return s


def break_into_batches(readers, args):
    '''
    :param readers: a dictionary {testsetname:reader}
    :return:
    '''
    test_data = []  # Tuples ("testname" : [ batch0_data, batch1_data, ...] ...)
    test_data_filenames = []  # Only output if batch size is 1
    for data_name,data_reader in readers.iteritems():
        tbatches = []
        tfilenames = []
        n_tbatches = int(math.ceil(len(data_reader.filenames) / (0.0 + args.max_test_batch_size)))
        for b in range(n_tbatches):
            start_b = b * args.max_test_batch_size
            end_b = min(len(data_reader.filenames), start_b + args.max_test_batch_size)
            data_in_batch = data_reader.get_specific_batch(range(start_b, end_b))
            tbatches.append(data_in_batch)
            if args.max_test_batch_size == 1:
                tfilenames.append(data_reader.orig_filenames[start_b])
            else:
                tfilenames.append('%s_batch%03d_%03d' % (data_name, start_b, end_b))
        test_data.append((data_name, tbatches))
        test_data_filenames.append((data_name, tfilenames))
        LOG.info(
            'TESTSETLOG Read "%s" test data %d of %d batches: %s' %
            (data_name, len(data_reader.filenames), len(tbatches), ' '.join([str(x.shape[0]) for x in tbatches])))

    test_set_names = [n for n,v in test_data]
    return test_data, test_set_names, test_data_filenames
