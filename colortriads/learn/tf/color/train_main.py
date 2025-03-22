#!/usr/bin/env python

import sys
import os
import math
import numpy as np
import random
import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import scipy.misc

import learn.tf.color.encoder as colorenc
import learn.tf.color.color_ops as color_ops
from learn.tf.log import LOG
import learn.tf.util as tf_util
import learn.tf.color.palette as palette
import learn.tf.color.setup_args
from learn.tf.color.data import RandomGrayscaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train color autoencoder')

    # DATA -----------------------------------------------------------------------
    parser.add_argument(
        '--traindata', action='store', type=str, required=True,
        help='File containing lines with location of data files.')
    parser.add_argument(
        '--viz_batch_size', action='store', type=int, default=20,
        help='First number of inputs in test sets to visualize; must be less than test batch size.')
    parser.add_argument(
        '--enable_viz_for_all_test_sets', action='store', type=bool, default=True,
        help='If true, will enable visualization for every test set; else just the first one.')

    # Data enrichment
    parser.add_argument(
        '--frac_bw_input', action='store', type=float, default=-1.0,
        help='If between 0 & 1, will randomly set this fraction of inputs to black and white.')
    parser.add_argument(
        '--add_color_glimpses', action='store', type=int, default=0,
        help='Number of color glimpses to add, if relevant.')

    # BASIC TRAINING OPS----------------------------------------------------------
    parser.add_argument(
        '--batch_size', action='store', type=int, default=20)
    parser.add_argument(
        '--its', action='store', type=int, default=1000,
        help='Number of iterations to run for.')

    # OPTIMIZER VARIABLES ------------------------------------------------------
    parser.add_argument(
        '--use_adam', action='store', type=bool, default=True)
    parser.add_argument(
        '--learning_rate', action='store', type=float, default=0.001)

    # SECONDARY SETTINGS --------------------------------------------------------

    parser.add_argument(
        '--tune_restored_palette_graph', action='store', type=bool, default=False,
        help='If this is false and --restore_palette_graph is set, will not tune ' +
        'the restored graph.')

    # SETTINGS SHARED BETWEEN TRAINING / TESTING -------------------------------
    learn.tf.color.setup_args.add_graph_flags(parser)
    learn.tf.color.setup_args.add_loss_flags(parser)
    learn.tf.color.setup_args.add_data_config_flags(parser)
    learn.tf.color.setup_args.add_testdata_flags(parser)
    learn.tf.color.setup_args.add_session_flags(parser)


    args = parser.parse_args()

    LOG.set_level(LOG.DEBUG)
    tf.reset_default_graph()

    # Initialize encoder ---------------------------------------------------------
    vae, popts = learn.tf.color.setup_args.setup_encoder_graph(args)

    # Set up data --------------------------------------------------------------------------------------------------
    train_files = args.traindata.strip().split(',')
    train_data = [ learn.tf.color.setup_args.setup_data_source(x, args) for x in train_files ]
    partial_input_masker = RandomGrayscaler(
        args.img_width, random_bw_fraction=args.frac_bw_input, color_glimpses=args.add_color_glimpses)
    for (f,td) in zip(train_files, train_data):
        LOG.info('Read train data %s: %d' % (f, len(td.filenames)))

    test_data = []  # Tuples ("testname" : [ batch0_input_dict], [batch1_input_dict] ...)
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
            tdict = { vae.x:data_in_batch }
            if vae.x_mask_input is not None:
                if vae.dropout is None:
                    tdict[vae.x_mask_input] = partial_input_masker.process_batch(data_in_batch,
                                                                                 do_mask=False)  # do not mask test input
                else:
                    tdict[vae.x_mask_input] = data_in_batch
            tbatches.append(tdict)
        test_data.append((data_name, tbatches))
        LOG.info(
            'TESTSETLOG Read "%s" test data %d of %d batches: %s' %
            (data_name, len(data_reader.filenames), len(tbatches), ' '.join([str(x) for x in tbatch_sizes])))
    test_set_names = [n for n,v in test_data]
    LOG.info('Processed %d test data sources' % len(test_files))

    # Initialize optimizer ---------------------------------------------------------------------------------------------
    loss, losses = learn.tf.color.setup_args.setup_loss(vae, args, test_summary_collections=test_set_names)

    with tf.name_scope('var_summaries'):
        tf_util.add_vars_summaries(collections=['train'])

    excluded_vars = []
    if len(args.restore_palette_graph) > 0 and not args.tune_restored_palette_graph:
        excluded_vars.append('palette_graph')

    optimizer = (tf.train.AdamOptimizer(args.learning_rate) if args.use_adam else
                 tf.train.GradientDescentOptimizer(args.learning_rate))
    clipped_grads, train = tf_util.get_clipped_grads(optimizer, loss, max_norm=10.0, exclude=excluded_vars)
    tf_util.add_grads_summaries(clipped_grads, collections=['train'])

    test_idx = range(args.viz_batch_size)
    viz_collections = ['test']
    if args.enable_viz_for_all_test_sets:
        viz_collections = test_set_names
    vae.init_visualization(range(len(test_idx)), test_summary_collections=viz_collections)

    # Set up helper and restore model ----------------------------------------------------------------------------------
    log_vars = [x for x in losses]
    log_vars.append(loss)
    sess, helper = learn.tf.color.setup_args.setup_session_and_helper(args, log_vars)

    # Save flags for reference (add random # in case more than one run)
    with open(os.path.join(args.run_dir, 'arguments_%d.txt' % random.randint(0,10000000)), 'w') as f:
        s = ' \\\n'.join(['--%s=%s' % (x[0], str(x[1]).replace('False', '')) for x in args.__dict__.items()]) + '\n'
        f.write(s)
        LOG.info(s)

    # Initialize and restore model, if present
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if len(args.restore_palette_graph) > 0:
        model_saver = tf.train.Saver(tf.trainable_variables('palette_graph'))
        model_saver.restore(sess, os.path.join(args.restore_palette_graph, 'model.ckpt'))
        LOG.info('Restored palette graph from: %s' % args.restore_palette_graph)

    helper.restore_model()

    LOG.info('All the variables ------------------------------------------------------------------------------------- ')
    tf_util.print_all_vars()
    LOG.info('All the trainable grads --------------------------------------------------------------------------------')
    for g,v in clipped_grads:
        LOG.info('Grad for %s: %s' % (v.name, str(g)))


    def print_grad_vals():
        grad_vals = sess.run([x[0] for x in clipped_grads], tdict)  # use last set test dict
        for i in range(len(grad_vals)):
            LOG.debug(
                'Gradient %s : %0.3f, %0.3f' % (clipped_grads[i][1].name, np.min(grad_vals[i]), np.max(grad_vals[i])))

    # Hack
    if args.encoder_mode == 3:
        alpha_val,palette_colors_val,total_loss_val = sess.run(
            [vae.rnn_steps[0]['alpha'], vae.rnn_steps[0]['palette'].colors, loss], tdict)  # use last set test dict
        LOG.info('step Alpha %s' % str(alpha_val.shape))
        LOG.info('step Palette colors %s' % str(palette_colors_val.shape))
        print(palette_colors_val)
        LOG.info('step Min alpha [%0.3f, %0.3f]' % (np.min(alpha_val), np.max(alpha_val)))
        LOG.info('Total Loss %s' % str(total_loss_val))
    # End of hack

    first_run = True
    for i in range(args.its):
        if args.debug_level > 1 or first_run:
            LOG.debug('Gradients for %d ---------------------------' % i)
            print_grad_vals()

        inputs = []
        for td in train_data:
            input, idx = td.get_random_batch(args.batch_size / len(train_data))
            inputs.append(input)
        input = np.concatenate(inputs, axis=0)
        input_dict =  { vae.x: input }
        if vae.x_mask_input is not None:
            if vae.dropout is None:
                input_dict[vae.x_mask_input] = partial_input_masker.process_batch(input)
            else:
                input_dict[vae.x_mask_input] = input

        helper.process_iteration(input_dict, test_data, force=first_run)
        sess.run(train, input_dict)
        first_run = i <= 3

    helper.process_iteration(input_dict, test_data, force=True)
