#!/usr/bin/env python

import sys
import os
import itertools
import numpy as np
import random
import argparse
import tensorflow as tf

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

    # BASIC TRAINING OPS----------------------------------------------------------
    parser.add_argument(
        '--batch_size', action='store', type=int, default=20)
    parser.add_argument(
        '--its', action='store', type=int, default=1000,
        help='Number of iterations to run for.')

    # OPTIMIZER VARIABLES ------------------------------------------------------

    parser.add_argument(
        '--disc_learning_rate', action='store', type=float, default=0.001)
    parser.add_argument(
        '--vae_learning_rate', action='store', type=float, default=0.001)
    parser.add_argument(
        '--gan_loss_fraction', action='store', type=float, default=0.5,
        help='Fraction of the loss that is assigned an adversarial component; weight of (1-gan_loss_fraction) ' +
        'will be assigned to the reconstruction loss.')

    parser.add_argument(
        '--train_discriminator_every_n', action='store', type=int, default=1,
        help='')
    parser.add_argument(
        '--train_encoder_every_n', action='store', type=int, default=1,
        help='')

    # SECONDARY SETTINGS --------------------------------------------------------
    parser.add_argument(
        '--restore_encoder_graph', action='store', type=str, default='',
        help='If set, will restore full UNET + palette graph variables from a checkpoint.')

    parser.add_argument(
        '--tune_restored_palette_graph', action='store', type=bool, default=False,
        help='If this is false and --restore_palette_graph is set, will not tune ' +
        'the restored graph.')

    # SHARED SETTINGS ------------------------------------------------------------
    learn.tf.color.setup_args.add_graph_flags(parser)
    learn.tf.color.setup_args.add_loss_flags(parser)
    learn.tf.color.setup_args.add_data_config_flags(parser)
    learn.tf.color.setup_args.add_testdata_flags(parser)
    learn.tf.color.setup_args.add_discriminator_flags(parser)
    learn.tf.color.setup_args.add_session_flags(parser)

    args = parser.parse_args()

    # Checks -----------------------------------------------------------------------------------------------------------
    if args.gan_loss_fraction > 1.0 or args.gan_loss_fraction < 0.0:
        raise RuntimeError('Error: --gan_loss_fraction must be in [0, 1]')

    # Prelimiaries -----------------------------------------------------------------------------------------------------
    LOG.set_level(LOG.DEBUG)
    tf.reset_default_graph()

    # Set up data sources ----------------------------------------------------------------------------------------------
    train_files = args.traindata.strip().split(',')
    train_data = [learn.tf.color.setup_args.setup_data_source(x, args) for x in train_files]
    for (f, td) in zip(train_files, train_data):
        LOG.info('Read train data %s: %d' % (f, len(td.filenames)))

    test_data = learn.tf.color.setup_args.read_testdata(args)
    test_set_names = [n for n, v in test_data]

    # Set encoder ----------------------------------------------------------------------------------------------------
    vae, popts = learn.tf.color.setup_args.setup_encoder_graph(args)

    # Standard losses
    recon_loss, recon_losses = learn.tf.color.setup_args.setup_loss(vae, args, test_summary_collections=test_set_names)

    # Set up image editing ---------------------------------------------------------------------------------------------

    # Only a fraction of images gets edited
    edit_idx = tf.placeholder(tf.int32, [None, 1], name="edit_idx")
    y_truth = tf.to_float(edit_idx)  # 1 - edited, 0 - not edited

    # Edits are provided at random at every iteration
    n_alphas = len(vae.rnn_steps)
    color_edits = tf.placeholder(tf.float32, [ n_alphas, None, popts.n_colors * popts.n_channels ], name="color_edits")
    wind_edits = None
    if popts.wind_nchannels > 0:
        nraw_ch = popts.wind_nchannels + 1 if popts.wind_nchannels > 1 else popts.wind_nchannels
        wind_edits = tf.placeholder(tf.float32, [ None, nraw_ch ], name="wind_edits")

    # Edited palettes
    new_palettes = []
    for i in range(n_alphas):  # Note: rnn_steps is a misnomer; currently just all the masks
        original_palette = vae.rnn_steps[i]['palette']

        LOG.info('Orig flat wind %d shape %s' % (i, str(original_palette.flat_wind.shape)))
        with tf.name_scope('edited_palette_%d' % i):
            new_flat_colors = tf.clip_by_value(
                original_palette.flat_colors + tf.squeeze(tf.gather(color_edits, [i]), axis=0) * y_truth, 0.0, 1.0)
            LOG.info('New wind_edits shape %s' % str(wind_edits.shape))

            new_flat_wind = tf.clip_by_value(original_palette.flat_wind + wind_edits * y_truth, -1.0, 1.0)  # Note: does not work for u-v

            edited_palette = palette.PaletteHelper(popts)
            edited_palette.init_deterministic_decoder(new_flat_colors, new_flat_wind)
            new_palettes.append(edited_palette)

    # Edited image
    with tf.name_scope('edited_image'):
        source_colors = tf.concat([tf.expand_dims(p.patch_colors, axis=1) for p in new_palettes], axis=1)
        recolored_images = tf.reshape(color_ops.compute_alpha_mask_reconstruction(
            vae.mapping, source_colors, target_alphas=vae.out_labels_flat, blend=True), tf.shape(vae.x))


    def generate_random_edits(batch_size):
        def _rand_sign():
            return 1.0 if random.random() < 0.5 else -1.0

        n_edits = int(random.uniform(batch_size * 0.3, batch_size * 0.7))
        edit_idx_val = [1 for x in range(n_edits - 1) ] + [ 0 for x in range(batch_size - n_edits - 1) ]
        random.shuffle(edit_idx_val)
        # Note: first 2 indices are always not-edited, edited, so that we can visualize
        edit_idx_val = np.array([0, 1] + edit_idx_val, dtype=np.int32)

        color_edits_val = np.zeros([n_alphas, batch_size, popts.n_colors * popts.n_channels])
        for b in range(batch_size):
            # Pick sails to edit
            n_sail_edits = random.randint(1, n_alphas)
            edited_sails = [True for x in range(n_sail_edits)] + [False for x in range(n_alphas - n_sail_edits)]
            random.shuffle(edited_sails)
            for s in range(n_alphas):
                if edited_sails[s]:
                    # Pick vertices to edit
                    n_vertex_edits = random.randint(1, popts.n_colors)
                    edited_vertices = [True for x in range(n_vertex_edits)] + [False for x in range(3 - n_vertex_edits)]
                    random.shuffle(edited_vertices)
                    for v in range(popts.n_colors):
                        if edited_vertices[v]:
                            for c in range(3):
                                color_edits_val[s, b, v * popts.n_channels + c] = _rand_sign() * np.random.normal(0.3, 0.2)
        wind_edits_val = None
        if popts.wind_nchannels > 0:
            nraw_ch = popts.wind_nchannels + 1 if popts.wind_nchannels > 1 else popts.wind_nchannels
            wind_edits_val = np.zeros([batch_size, nraw_ch], dtype=np.float32)

        return np.reshape(edit_idx_val, [batch_size, 1]), color_edits_val, wind_edits_val


    # Set up discriminator ---------------------------------------------------------------------------------------------
    disc = learn.tf.color.setup_args.setup_discriminator(args)
    disc.init_computed_image_input(args.img_width, recolored_images)
    disc.init_dcgan_graph(reuse=False)

    real_disc = learn.tf.color.setup_args.setup_discriminator(args)

    # Set up losses ----------------------------------------------------------------------------------------------------

    # Discriminator loss
    disc_loss_components, disc_loss = disc.init_discriminator_loss(y_truth)
    disc_loss = tf.identity(disc_loss, name='DISC_LOSS')

    # GAN loss (the loss on just the edited images)
    raw_gan_loss_components = disc.init_gan_loss_elements()
    gan_loss_components = tf.gather(raw_gan_loss_components, edit_idx)
    gan_loss = tf.reduce_sum(gan_loss_components) / tf.reduce_sum(y_truth)  # Normalize by num edited
    gan_loss = tf.identity(gan_loss, name='GAN_LOSS')

    # Total encoder loss
    vae_loss = (1.0 - args.gan_loss_fraction) * recon_loss + args.gan_loss_fraction * gan_loss
    vae_loss = tf.identity(vae_loss, name='TOTAL_VAE_LOSS')

    # Set up additional summaries --------------------------------------------------------------------------------------

    with tf.name_scope('var_summaries'):
        tf_util.add_vars_summaries(collections=['train'])

    for n in test_set_names + ['train']:
        with tf.name_scope('%s_loss' % n):
            tf.summary.scalar('DISC_LOSS', disc_loss, collections=[n])
            tf.summary.scalar('GAN_LOSS', gan_loss, collections=[n])
            tf.summary.scalar('TOTAL_VAE_LOSS', vae_loss, collections=[n])

    # Set up visualization ---------------------------------------------------------------------------------------------

    # Visualize mask learning on the text set
    test_idx = range(args.viz_batch_size)
    viz_collections = ['test']
    if args.enable_viz_for_all_test_sets:
        viz_collections = test_set_names
    vae.init_visualization(range(len(test_idx)), test_summary_collections=viz_collections)

    # Visualize training img, sails, all as one giant image:
    # orig_img  | masks
    # recon_img | palettes
    # edited_img| edited_palettes
    def create_visualization_for_idx(idx):
        orig_img = tf.expand_dims(vae.x[idx, ...], axis=0)
        recon_img = tf.expand_dims(vae.restored_image[idx, ...], axis=0)
        edited_img = tf.expand_dims(recolored_images[idx, ...], axis=0)
        masks = []
        pimgs = []
        edited_pimgs = []
        for s in range(n_alphas):
            step_alpha = tf.slice(vae.out_labels,
                                  [idx, 0, 0, s],
                                  [1, vae.img_width, vae.img_width, 1])
            masks.append(tf.tile(step_alpha, [1, 1, 1, 3]))
            pimgs.append(tf.expand_dims(vae.rnn_steps[s]['palette'].get_viz_for_idx_py_func(idx, vae.img_width), axis=0))
            edited_pimgs.append(tf.expand_dims(new_palettes[s].get_viz_for_idx_py_func(idx, vae.img_width), axis=0))

        img = tf.concat([tf.concat([orig_img] + masks, axis=1),
                         tf.concat([recon_img] + pimgs, axis=1),
                         tf.concat([edited_img] + edited_pimgs, axis=1)], axis=2)
        return color_ops.to_uint8(img)


    with tf.name_scope('TRAIN_GAN_VIZ'):
        tf.summary.image('untouched_img', create_visualization_for_idx(0), collections=['train'])
        tf.summary.image('edited_img', create_visualization_for_idx(1), collections=['train'])


    # Save flags for reference (add random # in case more than one run)
    with open(os.path.join(args.run_dir, 'arguments_%d.txt' % random.randint(0, 10000000)), 'w') as f:
        s = ' \\\n'.join(['--%s=%s' % (x[0], str(x[1]).replace('False', '')) for x in args.__dict__.items()]) + '\n'
        f.write(s)
        LOG.info(s)

    # Set up encoder optimizer -----------------------------------------------------------------------------------------

    palette_vars = [ 'palette_graph' ]
    vae_vars = [ 'UNETdecoder_', 'encoder_' ]
    vae_clipped_grads = None
    vae_trainable = vae_vars + (palette_vars if args.tune_restored_palette_graph else [])
    vae_optimizer = tf.train.AdamOptimizer(args.vae_learning_rate)
    vae_clipped_grads, vae_train = tf_util.get_clipped_grads(vae_optimizer, vae_loss, max_norm=10.0, include=vae_trainable)
    tf_util.add_grads_summaries(vae_clipped_grads, collections=['train'])

    # Set up discriminator optimizer -----------------------------------------------------------------------------------

    disc_vars = [ disc.strid ]
    disc_optimizer = tf.train.AdamOptimizer(args.disc_learning_rate)
    disc_clipped_grads, disc_train = tf_util.get_clipped_grads(disc_optimizer, disc_loss, max_norm=10.0, include=disc_vars)
    tf_util.add_grads_summaries(disc_clipped_grads, collections=['train'])

    # Session / restoring ----------------------------------------------------------------------------------------------

    log_vars = [x for x in recon_losses] + [gan_loss, vae_loss, disc_loss]
    sess, helper = learn.tf.color.setup_args.setup_session_and_helper(args, log_vars)

    # Initialize and restore model, if present
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Restore unet model if supplied
    if len(args.restore_encoder_graph) > 0:
        saved_vars = [ tf.trainable_variables(x) for x in vae_vars + palette_vars ]
        saved_vars = list(itertools.chain.from_iterable(saved_vars))
        model_saver = tf.train.Saver(saved_vars)

        model_saver.restore(sess, os.path.join(args.restore_encoder_graph, 'model.ckpt'))
        LOG.info('Restored encoder graph from: %s' % args.restore_encoder_graph)

    helper.restore_model()

    LOG.info('All the variables ------------------------------------------------------------------------------------- ')
    tf_util.print_all_vars()
    LOG.info('All the (vae) trainable grads ------------------------------------------------------------------------- ')
    if vae_clipped_grads is not None:
        for g, v in vae_clipped_grads:
            LOG.info('Grad for %s: %s' % (v.name, str(g)))
    LOG.info('All the (discriminator) trainable grads --------------------------------------------------------------- ')
    for g, v in disc_clipped_grads:
        LOG.info('Grad for %s: %s' % (v.name, str(g)))

    # Training ---------------------------------------------------------------------------------------------------------
    def make_input_dict(batch_data):
        edit_idx_val, color_edits_val, wind_edits_val = generate_random_edits(batch_data.shape[0])

        res = { vae.x: batch_data,
                vae.x_mask_input: batch_data,
                edit_idx: edit_idx_val,
                color_edits: color_edits_val }

        if wind_edits_val is not None:
            res[wind_edits] = wind_edits_val
        return res

    first_run = True
    for i in range(args.its):
        # Get inputs from all train sets
        inputs = []
        for td in train_data:
            input, idx = td.get_random_batch(args.batch_size / len(train_data))
            inputs.append(input)
        input = np.concatenate(inputs, axis=0)

        # Set up random edits
        input_dict = make_input_dict(input)
        test_dicts = [ (x[0], [make_input_dict(y) for y in x[1]]) for x in test_data ]
        #print(test_dicts[0][1][0])
        # Log/save
        helper.process_iteration(input_dict, test_dicts, force=first_run)
        first_run = i <= 3

        # Train
        if args.train_encoder_every_n > 0 and (i % args.train_encoder_every_n) == 0:
            sess.run(vae_train, input_dict)

        if args.train_discriminator_every_n > 0 and (i % args.train_discriminator_every_n) == 0:
            sess.run(disc_train, input_dict)

    helper.process_iteration(input_dict, test_data, force=True)



    # Options:
    # (What to train)
    # - train discriminator
    # - train encoder
    # - train palette graph
    # - train any combination of these guys
    # (Recon losses)
    # (Restoring)
    # - restore unet (and palette) graph from y
    # - restore everything from x
