#!/usr/bin/env python

import sys
import os
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import skimage.io

import learn.tf.color.encoder as colorenc
import learn.tf.color.color_ops as color_ops
from learn.tf.log import LOG
import learn.tf.util as tf_util
import learn.tf.color.palette as palette
import learn.tf.color.setup_args
from learn.tf.color.data import ColorGlimpsesDataHelper

# eval_metrics_main:
# - input: testset, model location, config flags
# - output: testsetlog just as from the run for a set of custom metrics (per batch)

# run_interact_main:
# - input: model location, config flags
# - interactive input: image location
# - interactive output: written palette, written alphas (opt), error values,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train color autoencoder')

    # DATA -----------------------------------------------------------------------
    parser.add_argument(
        '--testdata', action='store', type=str, required=True,
        help='File containing lines with location of data files.')
    parser.add_argument(
        '--max_test_batch_size', action='store', type=int, default=1,
        help='Sets the maximum number of test values to process concurrently; if testdata ' +
        'is bigger, will break it up')
    parser.add_argument(
        '--verbose_output', action='store', type=bool, default=True,
        help='If true, dumps a lot of the visualization files, else just the binary and image.')

    # BASIC TRAINING OPS----------------------------------------------------------
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True,
        help='Dir where to output results.')
    parser.add_argument(
        '--run_dir', action='store', type=str, required=True,
        help='Directory where to put the model checkpoints, etc.')
    parser.add_argument(
        '--max_outputs', action='store', type=int, default=50,
        help='Maximum number of files to provide file output of, in the form of binary files ' +
        'and visualization')

    # SETTINGS SHARED BETWEEN TRAINING / TESTING -------------------------------
    print('TIMING RUN: Start: %0.5f' % time.time())
    learn.tf.color.setup_args.add_graph_flags(parser)
    learn.tf.color.setup_args.add_loss_flags(parser, required=False)
    learn.tf.color.setup_args.add_data_config_flags(parser)

    args, unknown = parser.parse_known_args()
    print(args)

    # HACK:
    #args.img_width = 512
    #args.patch_width = -1
    #args.input_mode = 2
    #args.max_tri_subdivs=16

    LOG.set_level(LOG.DEBUG)
    tf.reset_default_graph()
    LOG.info(learn.tf.color.setup_args.create_argument_string(args))

    # Initialize encoder ---------------------------------------------------------
    vae, popts = learn.tf.color.setup_args.setup_encoder_graph(args)

    test_files = args.testdata.strip().split(',')
    data_readers = learn.tf.color.setup_args.setup_data_sources(test_files, args, is_test=True)  # {testname: reader}
    tbatches, test_set_names, tbatch_filenames = learn.tf.color.setup_args.break_into_batches(data_readers, args)
    tbatch_filenames = dict(tbatch_filenames)

    # Build test dicts:
    def _build_td(x):
        res = { vae.x : x }
        if vae.x_mask_input is not None:
            res[vae.x_mask_input] = x
        return res

    test_data = [(k, [ _build_td(b) for b in vals ]) for k,vals in tbatches]
    loss, losses = learn.tf.color.setup_args.setup_loss(vae, args, test_summary_collections=test_set_names)

    sess = tf.Session()
    vae.init_visualization([0],test_summary_collections=test_set_names)  # we only visualize 1st element of a batch

    # Initialize and restore model, if present
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if len(args.restore_palette_graph) > 0:
        model_saver = tf.train.Saver(tf.trainable_variables('palette_graph'))
        model_saver.restore(sess, os.path.join(args.restore_palette_graph, 'model.ckpt'))
        LOG.info('Restored palette graph from: %s' % args.restore_palette_graph)

    # This one just restores the model vars
    helper = tf_util.TrainHelper(args.run_dir, sess, save_summary_every_n=100,
                                 overwrite=False, log_vars=losses + [loss], log_every_n=1)
    helper.restore_model()


    LOG.info('All the variables ------------------------------------------------------------------------------------- ')
    tf_util.print_all_vars()
    print('TIMING RUN: Model loaded: %0.5f' % time.time())

    os.makedirs(args.output_dir)
    if args.max_test_batch_size > 1:
        LOG.info('Evaulate (batched) ------------------------------------------------------------------------------------ ')
        helper.evaluate_test_batches(test_data)
    else:
        LOG.info('Evaulate (file level) --------------------------------------------------------------------------------- ')
        with open(os.path.join(args.output_dir, 'file_losses.txt'), 'w') as lf:
            helper.evaluate_all_files(data_readers, _build_td, lf)
    print('TIMING RUN: Evaluation done: %0.5f' % time.time())

    LOG.info('Process outputs --------------------------------------------------------------------------------------- ')

    for n in range(len(tbatches)):
        testname, tb = tbatches[n]
        tbn = tbatch_filenames[testname]
        _, testdicts = test_data[n]

        output_subdir = os.path.join(args.output_dir, 'results_%s' % testname)
        if not os.path.exists(output_subdir):
            LOG.info('Making subdirectory: %s' % output_subdir)
            os.makedirs(output_subdir)
        if args.verbose_output:
            verbose_output_subdir = os.path.join(output_subdir, 'verbose')
            if not os.path.exists(verbose_output_subdir):
                os.makedirs(verbose_output_subdir)
        else:
            verbose_output_subdir = None

        # Output one visualization per batch ? Or everything ?
        if args.max_outputs >= 0:
            nout = min(len(tb), args.max_outputs)
        else:
            nout = len(tb)

        for i in range(nout):
            prefix = '.'.join(os.path.basename(tbn[i]).split('.')[0:-1])   #'%s_%05d' % (testname, i)
            verbose_output_prefix = None
            if verbose_output_subdir:
                verbose_output_prefix = os.path.join(verbose_output_subdir, prefix)

            test_dict = testdicts[i]
            vae.eval_write_outputs(
                os.path.join(output_subdir, prefix), sess, test_dict, verbose_output_prefix)

            if verbose_output_prefix:
                res = tf_util.evaluate_var_dict(sess, vae.result_vars, test_dict)
                viz = tf_util.evaluate_var_dict(sess, vae.result_viz_vars, test_dict)
                LOG.info('Evaluated result keys: %s' % str(res.keys()))
                LOG.info('Evaluated viz keys: %s' % str(viz.keys()))

                for k,v in viz.iteritems():
                    LOG.info('%s: %s' % (k, str(v.shape)))
                    skimage.io.imsave('%s_viz_%s.png' % (verbose_output_prefix, k), v[0])

                for k,v in res.iteritems():
                    LOG.info('Value of %s: shape = %s' % (k, v.shape))
                    LOG.info('%0.3f, %0.3f (%d nonzero, %0.2f sum)' %
                            (np.min(v), np.max(v), np.count_nonzero(v), np.sum(v)))
    LOG.info('Results in %s' % args.output_dir)
