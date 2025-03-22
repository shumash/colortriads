#!/usr/bin/env python

import os
import psutil
import time
import argparse
import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from learn.tf.log import LOG
import learn.tf.util as tf_util
import learn.tf.color.setup_args

import learn.tf.color.encoder as colorenc
import learn.tf.color.color_ops as color_ops
import learn.tf.color.palette as palette
from learn.tf.color.data import ColorGlimpsesDataHelper


if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    print('Start {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))

    parser = argparse.ArgumentParser()

    parser.add_argument(
    	'--input', action='store', type=str, required=True,
    	help='Image file or ')
    parser.add_argument(
    	'--do_map', action='store_true')
    parser.add_argument(
    	'--output_prefix', action='store', type=str, required=True)

    learn.tf.color.setup_args.add_graph_flags(parser)
    learn.tf.color.setup_args.add_loss_flags(parser, required=False)

    args, unknown = parser.parse_known_args()
    print(args)

    LOG.set_level(LOG.DEBUG)
    tf.reset_default_graph()
    LOG.info(learn.tf.color.setup_args.create_argument_string(args))

    print('TF cleared {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))

    # Initialize encoder ---------------------------------------------------------
    vae, popts = learn.tf.color.setup_args.setup_encoder_graph(args)
    loss, losses = learn.tf.color.setup_args.setup_loss(vae, args)

    for v in tf.global_variables():
        print('VAR: %s: %s' % (str(v.dtype), str([x.value for x in v.shape])))

    print('VAE created {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))
    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print('Sess created {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))

    if len(args.restore_palette_graph) > 0:

        reader = tensorflow.train.load_checkpoint(
            os.path.join(args.restore_palette_graph, 'model.ckpt'))
        read_vars = reader.get_variable_to_shape_map()

        assign_ops = []
        for var in tf.trainable_variables('palette_graph'):
            name = var.name.split(':')[0]  # Remove the ':0' suffix
            name = name.replace("/fc", "/varfc")
            name = name.replace("kernel", "weights").replace("bias", "biases")
            name = name.replace("/dense_1", "")
            name = name.replace("/dense_2", "")
            name = name.replace("/dense_3", "")
            name = name.replace("/dense", "")

            print(f'{var.name}: {name} --> {var.shape}')
            value = reader.get_tensor(name)
            print(f'Read variable: {value.shape}')

            assign_op = tf.assign(var, value)
            assign_ops.append(assign_op)

        sess.run(assign_ops)
  
        LOG.info('Restored palette graph from: %s' % args.restore_palette_graph)

    tf_util.print_all_vars()
    print('TIMING RUN: Model loaded: %0.5f' % time.time())

    print('Model loaded {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))

    data_reader = learn.tf.color.setup_args.single_image_data_source(args.input, args.img_width)
    input_dict = { vae.x: data_reader.get_all() }
    print('Input read {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))
    if vae.x_mask_input is not None:
            input_dict[vae.x_mask_input] = input_dict[vae.x]

    vae.eval_write_outputs(
                args.output_prefix, sess, input_dict, verbose_output_prefix=None)
    print('End {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))
