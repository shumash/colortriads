import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.keras.layers import Dense
import numpy as np
import base64
import os
import psutil
import time
from skimage.io import imsave

import vgg16
from learn.tf.log import LOG
import learn.tf.color.data as DATA
import learn.tf.color.color_ops as color_ops
import learn.tf.color.losses as losses
import learn.tf.color.palette as palette
import learn.tf.util as tf_util
import util.img_util as img_util


# UTILITIES ------------------------------------------------------------------------------------------------------------

def log_tensor(t, name):
    print('{}: {} {}'.format(name, t.dtype, t.shape))

def read_hist_to_row(filename):
    return DATA.read_hist_to_row(filename)

def make_palette_dict_binary_output(colors, patchwork, wind=None):
    return { 'colors': colors.reshape([3,3]), 'patchwork': patchwork, 'wind': (wind if wind is None else wind.reshape(-1)[0:3])}

def write_binary_color_sail_rig_file(fname, input_img, palette_dicts, uv_mappings, alphas=None):  #colors, patchwork, wind=None):
    '''
    Writes color sail rig to a binary file (compatible with javascript UI).

    :param fname:
    :param input_img:
    :param vals:
    :param palettes:
    :param uv_mappings:
    :param alphas:
    :return:
    '''
    def _pvals_to_data(p, pv, i=0):
        wind = None
        if ('wind%d' % i) in pv:
            wind = pv['wind%d' % i][0]
        return palette.color_sail_to_float32_arr(pv['colors%d' % i][0], patchwork=p.opts.max_tri_subdivs, wind=wind)

    width = input_img.shape[0]
    nalphas = len(palette_dicts)
    info_data = np.array([nalphas, width, 0, 0], np.int16)

    if input_img.shape[1] != width:
        raise RuntimeError('Non-square image binary I/O not supported')

    if input_img.dtype != np.uint8:  # ensure uint
        input_img = (input_img * 255).astype(np.uint8)

    if len(input_img.shape) == 2 or input_img.shape[2] == 1:  # ensure color
        input_img = np.concatenate([input_img.reshape([width, width, 1]) for x in range(3)], axis=2)

    if input_img.shape[2] == 3: # ensure RGBA
        input_img = np.concatenate([input_img, np.ones([width, width, 1], np.uint8) * 255], axis=2)

    img_data = np.reshape(input_img, [-1])
    mappings_data = np.concatenate([np.reshape(x, [-1]) for x in uv_mappings])

    if alphas is None:
        mock_alpha = np.ones([width, width], np.float32)
        alphas = [mock_alpha for x in range(nalphas)]
    alphas_data = np.concatenate([np.reshape(x, [-1]) for x in alphas])

    # print ('Total data size %s %s' % (str(uint_data.shape), str(float_data.shape)))
    # total_data = np.array([1, 0.5, 6.7, 3.3], np.float32)
    with open(fname, 'wb') as f:
        # Info
        f.write(info_data.tobytes())
        # Palette
        for p in palette_dicts:
            f.write(palette.color_sail_to_float32_arr(p['colors'], p['patchwork'], p['wind']).tobytes())
        # Image
        f.write(img_data.tobytes())
        # Mappings
        f.write(mappings_data.tobytes())
        # Alphas
        f.write(alphas_data.tobytes())


def eval_write_outputs(self, output_prefix, sess, input_dict, verbose_output_prefix):
        '''
        Evaluates and writes the model on the input dict, and writes the outputs for the first element in
        the batch only. Is intended for use with a batch size of 1.
        '''
        def _get_pvars(p, i=0):
            vars = { ('colors%d' % i) : p.colors }
            if p.wind is not None:
                vars['wind%d' % i] = p.wind
            return vars

        def _pvals_to_str(p, pv, i=0):
            wind = None
            if ('wind%d' % i) in pv:
                wind = pv['wind%d' % i][0]
            return palette.color_sail_to_string(
                pv['colors%d' % i][0], patchwork=p.opts.max_tri_subdivs, wind=wind)

        def _pvals_to_data(p, pv, i=0):
            wind = None
            if ('wind%d' % i) in pv:
                wind = pv['wind%d' % i][0]
            return palette.color_sail_to_float32_arr(pv['colors%d' % i][0], patchwork=p.opts.max_tri_subdivs, wind=wind)

        def _mapping_to_img(p, mapping, i=0):
            # This is more correct: color_idx_to_color_bary, but we don't use it b/c during recoloring
            # pixels read from the palette will be anti-aliased at the edges and so produce a faulty output
            uv = p.color_idx_to_center_bary(mapping[0, :, i, 2])
            uv_img = np.concatenate([np.reshape(uv, [self.img_width, self.img_width, 2]),
                                     np.zeros([self.img_width, self.img_width, 1])], axis=2)
            return uv, uv_img

        def _mapping_to_str(uv):
            return (' '.join([str(x) for x in uv[:, 0]]) + '\n' +
                    ' '.join([str(x) for x in uv[:, 1]]) + '\n')

        def _mappings_encode(data):
            '''Data: [Npixels, Npalettes, 2]'''
            array = np.reshape(data, [-1])
            return base64.b64encode(array)

        def _write_binary_file(fname, input_img, vals, palettes, uv_mappings, alphas):
            width = input_img.shape[0]
            nalphas = len(alphas)
            info_data = np.array([nalphas, width, 0, 0], np.int16)
            img_data = np.reshape(  # RGBA
                np.concatenate([(input_img * 255).astype(np.uint8),
                                np.ones([width, width, 1], np.uint8) * 255], axis=2), [-1])
            mappings_data = np.concatenate([np.reshape(x, [-1]) for x in uv_mappings])
            alphas_data = np.concatenate([np.reshape(x, [-1]) for x in alphas])

            # print ('Total data size %s %s' % (str(uint_data.shape), str(float_data.shape)))
            # total_data = np.array([1, 0.5, 6.7, 3.3], np.float32)
            with open(fname, 'w') as f:
                # Info
                f.write(info_data.tobytes())
                # Palettes
                for i in range(len(palettes)):
                    f.write(_pvals_to_data(palettes[i], vals, i).tobytes())
                # Image
                f.write(img_data.tobytes())
                # Mappings
                f.write(mappings_data.tobytes())
                # Alphas
                f.write(alphas_data.tobytes())


        # Get all the vals we need to evaluate
        vars = { 'input' : self.x }
        if self.mapping is not None:
            vars['mapping'] = self.mapping

        if self.palette is not None:
            vars.update(_get_pvars(self.palette))

            vals = tf_util.evaluate_var_dict(sess, vars, input_dict)
            with open(output_prefix + '_palette.txt', 'w') as f:
                f.write(_pvals_to_str(self.palette, vals) + '\n')

            if 'mapping' in vals:
                mapping = np.expand_dims(vals['mapping'], axis=2)
                mapping_uv, mapping_uv_img = _mapping_to_img(self.palette, mapping, 0)

                mock_alpha = np.ones([self.img_width, self.img_width], np.float32)
                _write_binary_file(output_prefix + '_RES.binary',
                                   vals['input'][0],
                                   vals,  # for info for the palette
                                   palettes=[self.palette],
                                   uv_mappings=[mapping_uv],
                                   alphas=[mock_alpha])


class ColorAutoencoder(object):
    '''
    Object that can be configured with a number of networks for training a color
    palette encoder.
    '''
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.x = None
        self.batch_size = None
        self.flat_hist = None
        self.layers = []

        # Utilities for running
        self.result_vars = {}
        self.result_viz_vars = {}

        # Palette
        self.encoding = None
        self.levels = None
        self.wind = None
        self.palette = None
        self.palette_colors = None   # All interpolated colors from one or multiple palettes
        self.wind = None

        # [Nbatches x Npalettes x ncolors x 3]
        # These are all the blended palette colors, not the vertex colors
        self.palettes_colors = None
        self.lab_palette_colors = None

        # Specific to image input
        self.img_colors = None
        self.lab_img_colors = None
        self.img_width = None
        self.patch_width = None
        self.hist = None
        self.hist_count_info = None

        # Masks
        self.out_labels = None
        self.out_labels_flat = None
        self.x_mask_input = None  # Note: different from actual input x which has full data used for loss
        self.restored_image = None
        self.mapping = None
        self.out_labels_viz = None

        # RNN
        self.rnn_steps = None


    # OUTPUT ---------------------------------------------------------------------------------------
    def eval_write_outputs(self, output_prefix, sess, input_dict, verbose_output_prefix):
        process = psutil.Process(os.getpid())
        '''
        Evaluates and writes the model on the input dict, and writes the outputs for the first element in
        the batch only. Is intended for use with a batch size of 1.
        '''
        def _get_pvars(p, i=0):
            vars = { ('colors%d' % i) : p.colors }
            if p.wind is not None:
                vars['wind%d' % i] = p.wind
            return vars

        def _pvals_to_str(p, pv, i=0):
            wind = None
            if ('wind%d' % i) in pv:
                wind = pv['wind%d' % i][0]
            return palette.color_sail_to_string(
                pv['colors%d' % i][0], patchwork=p.opts.max_tri_subdivs, wind=wind)

        def _pvals_to_data(p, pv, i=0):
            wind = None
            if ('wind%d' % i) in pv:
                wind = pv['wind%d' % i][0]
            return palette.color_sail_to_float32_arr(pv['colors%d' % i][0], patchwork=p.opts.max_tri_subdivs, wind=wind)

        def _mapping_to_img(p, mapping, i=0):
            # This is more correct: color_idx_to_color_bary, but we don't use it b/c during recoloring
            # pixels read from the palette will be anti-aliased at the edges and so produce a faulty output
            uv = p.color_idx_to_center_bary(mapping[0, :, i, 2])
            uv_img = np.concatenate([np.reshape(uv, [self.img_width, self.img_width, 2]),
                                     np.zeros([self.img_width, self.img_width, 1])], axis=2)
            return uv, uv_img

        def _mapping_to_str(uv):
            return (' '.join([str(x) for x in uv[:, 0]]) + '\n' +
                    ' '.join([str(x) for x in uv[:, 1]]) + '\n')

        def _mappings_encode(data):
            '''Data: [Npixels, Npalettes, 2]'''
            array = np.reshape(data, [-1])
            return base64.b64encode(array)

        def _write_binary_file(fname, input_img, vals, palettes, uv_mappings, alphas):
            width = input_img.shape[0]
            nalphas = len(alphas)
            info_data = np.array([nalphas, width, 0, 0], np.int16)
            img_data = np.reshape(  # RGBA
                np.concatenate([(input_img * 255).astype(np.uint8),
                                np.ones([width, width, 1], np.uint8) * 255], axis=2), [-1])
            mappings_data = np.concatenate([np.reshape(x, [-1]) for x in uv_mappings])
            alphas_data = np.concatenate([np.reshape(x, [-1]) for x in alphas])

            # print ('Total data size %s %s' % (str(uint_data.shape), str(float_data.shape)))
            # total_data = np.array([1, 0.5, 6.7, 3.3], np.float32)
            with open(fname, 'wb') as f:
                # Info
                f.write(info_data.tobytes())
                # Palettes
                for i in range(len(palettes)):
                    f.write(_pvals_to_data(palettes[i], vals, i).tobytes())
                # Image
                f.write(img_data.tobytes())
                # Mappings
                f.write(mappings_data.tobytes())
                # Alphas
                f.write(alphas_data.tobytes())


        # Get all the vals we need to evaluate
        vars = { 'input' : self.x }
        if self.mapping is not None:
            vars['mapping'] = self.mapping

        if self.palette is not None:
            vars.update(_get_pvars(self.palette))

            start_time = time.time()
            #print('Middle {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))
            vals = tf_util.evaluate_var_dict(sess, vars, input_dict)
            #print('Middle {} MB'.format(process.memory_info()[0] / 1000.0 / 1000.0))
            total_time = time.time() - start_time
            with open(output_prefix + '.eval_time.txt', 'w') as f:
                f.write('%0.5f seconds\n' % total_time)
            with open(output_prefix + '.palette.txt', 'w') as f:
                f.write(_pvals_to_str(self.palette, vals) + '\n')

            if 'mapping' in vals:
                mapping = np.expand_dims(vals['mapping'], axis=2)
                mapping_uv, mapping_uv_img = _mapping_to_img(self.palette, mapping, 0)

                mock_alpha = np.ones([self.img_width, self.img_width], np.float32)
                _write_binary_file(output_prefix + '_RES.binary',
                                   vals['input'][0],
                                   vals,  # for info for the palette
                                   palettes=[self.palette],
                                   uv_mappings=[mapping_uv],
                                   alphas=[mock_alpha])
                if verbose_output_prefix:
                    log_tensor(mapping_uv_img, 'mapping_uv_img')
                    imsave(verbose_output_prefix + '_mapping.bmp', mapping_uv_img)

                    with open(verbose_output_prefix + '_mapping.txt', 'w') as f:
                        f.write(_mapping_to_str(mapping_uv))

        elif self.rnn_steps is not None:

            for i in range(len(self.rnn_steps)):
                step = self.rnn_steps[i]
                vars.update(_get_pvars(step['palette'], i))
            vars['alphas'] = self.out_labels

            # Evaluate
            vals = tf_util.evaluate_var_dict(sess, vars, input_dict)

            # Write all the palettes to one file
            with open(output_prefix + '_palette.txt', 'w') as f:
                for i in range(len(self.rnn_steps)):
                    step = self.rnn_steps[i]
                    f.write(_pvals_to_str(step['palette'], vals, i) + '\n')

            alpha_argmax = np.argmax(vals['alphas'][0, :, :, :], axis=2)
            nalphas = vals['alphas'].shape[-1]
            binary_alphas = np.zeros([vals['alphas'].shape[1], vals['alphas'].shape[2], nalphas], np.float32)
            for i in range(nalphas):
                binary_alphas[:,:,i][alpha_argmax == i] = 1.0

            # Write all the mappings to images
            mappings = []
            for i in range(len(self.rnn_steps)):
                step = self.rnn_steps[i]
                uv, uv_img = _mapping_to_img(step['palette'], vals['mapping'], i)
                alpha_bin = np.expand_dims(binary_alphas[:, :, i], axis=2)

                #print('UV size: %s' % str(uv.shape))
                #print('UV img: %s' % str(uv_img.shape))
                #print('Alpha : %s' % str(alpha_bin.shape))
                a_img = uv_img * alpha_bin  # weigh by alpha to make meaningful
                mappings.append((uv, uv_img, a_img))

                if verbose_output_prefix:
                    with open(verbose_output_prefix + ('_mapping%d.txt' % i), 'w') as f:
                        f.write(_mapping_to_str(uv))

                    imsave(verbose_output_prefix + ('_mapping%d.bmp' % i), uv_img)
                    imsave(verbose_output_prefix + ('_alpha_bin%d.bmp' % i), np.squeeze(alpha_bin))
                    imsave(verbose_output_prefix + ('_alpha%d.bmp' % i), vals['alphas'][0,:,:,i])


            if verbose_output_prefix:
                mappings_img = np.concatenate([x[2] for x in mappings], axis=0)
                imsave(verbose_output_prefix + '_mappings.bmp', mappings_img)

            #print('Mapping: %s' % str(vals['mapping'].shape))
            #print('Alphas: %s ' % str(vals['alphas'].shape))

            _write_binary_file(output_prefix + '_RES.binary',
                               vals['input'][0],
                               vals,  # for info for all palettes
                               palettes=[s['palette'] for s in self.rnn_steps],
                               uv_mappings=[x[0] for x in mappings],
                               alphas = [vals['alphas'][0][:, :, x] for x in range(nalphas)])

        #imsave('%s_input.png' % output_prefix, vals['input'][0])


    # INPUT ----------------------------------------------------------------------------------------
    def init_hist_input(self):
        if self.x is not None:
            raise RuntimeError('ColorAutoencoder input can only be initialized once')

        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, [None, self.n_bins * self.n_bins * self.n_bins], name="vae_input")
            self.batch_size = tf.shape(self.x)[0]
        self.flat_hist = self.x


    def init_image_input(self, img_width):
        '''
        Use this with the init_image_encoder_graph function if histogram is not used for loss.
        :param img_width:
        :return:
        '''
        if self.x is not None:
            raise RuntimeError('ColorAutoencoder input can only be initialized once')

        self.img_width = img_width

        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.float32, [None, img_width, img_width, 3], name="vae_input")
            self.batch_size = tf.shape(self.x)[0]
            self.img_colors = tf.reshape(self.x, [self.batch_size, -1, 3])
            self.lab_img_colors = color_ops.rgb2lab_anyshape(self.img_colors)

    # INPUT PROCESSING -----------------------------------------------------------------------------

    def add_histogram_computation(self, patch_width=-1, zero_out_white=False):
        if self.flat_hist is not None:
            LOG.warn('Input histogram already initialized')
            return

        with tf.name_scope('img_to_histogram'):
            if patch_width < 0:
                self.hist, self.flat_hist, hist_vars = color_ops.compute_hist(
                      self.img_colors, self.n_bins, normalize=True, zero_out_white=zero_out_white, squeeze=True)
            else:
                LOG.info('Initializing patch histograms')
                self.patch_width = patch_width
                hist_vars = color_ops.compute_patch_hist(
                    self.x, hist_subdivs=self.n_bins, patch_width=self.patch_width,
                    zero_out_white=zero_out_white)
                self.hist = hist_vars['hist3d']
                self.flat_hist = hist_vars['norm_max_hist']
            self.hist_count_info = hist_vars['count_info']  # To compute weighted histogram later


    # ----------------------------------------------------------------------------------------------
    # NETWORKS -------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    # RNN V1 ---------------------------------------------------------------------------------------
    def init_vgg_masks_graph(self,
        input_channels,
        palette_graph_arch,
        palette_opts,
        npalettes,
        use_rnn,
        palette_layer_specs,
        mask_softmax_temp,
        dropout=None):

        if dropout is not None:
            raise RuntimeError('Cannot use dropout mode with VGG!')

        self.init_mask_graph(input_channels, palette_graph_arch, palette_opts=palette_opts,
                             npalettes=npalettes, use_vgg=True,
                             use_rnn=use_rnn, palette_layer_specs=palette_layer_specs,
                             mask_softmax_temp=mask_softmax_temp)


    def init_vanilla_masks_graph(self,
        input_channels,
        palette_graph_arch,
        palette_opts,
        npalettes,
        use_rnn,
        palette_layer_specs,
        mask_softmax_temp,
        dropout=None):

        self.init_mask_graph(input_channels, palette_graph_arch, palette_opts=palette_opts,
                             npalettes=npalettes, use_vgg=False,
                             use_rnn=use_rnn, palette_layer_specs=palette_layer_specs,
                             mask_softmax_temp=mask_softmax_temp, dropout=dropout)

    def init_mask_graph(self, input_channels, palette_graph_arch, palette_opts,
                        npalettes, use_vgg, use_rnn, palette_layer_specs, featuremap_filters=64,
                        mask_softmax_temp=1.0, dropout=None):
        if self.img_width is None:
            raise RuntimeError('Must have image input to encode masks')


        self.dropout = dropout
        self.x_mask_input = tf.placeholder(tf.float32, [None, self.img_width, self.img_width, input_channels], name="vae_mask_input")
        fmap_filters = featuremap_filters if use_rnn else npalettes

        # Step1: init vgg encoder
        if use_vgg:
            self.fmap = self.init_vgg_unet(output_filters=fmap_filters)
        else:
            self.fmap = self.init_vanilla_unet(output_filters=fmap_filters,
                dropout=dropout)

        if use_rnn:
            self.init_rnn_masks(npalettes, palette_graph_arch, palette_opts=palette_opts,
                                palette_layer_specs=palette_layer_specs, fmap_filters=fmap_filters)
        else:
            self.init_oneshot_masks(npalettes, palette_graph_arch, palette_opts=palette_opts,
                                    palette_layer_specs=palette_layer_specs,
                                    mask_softmax_temp=mask_softmax_temp)

    def init_rnn_masks(self, seq_length, palette_graph_arch, palette_opts, palette_layer_specs, fmap_filters):
        if self.fmap is None:
            raise RuntimeError('Rnn requires fmap')

        global_mask = tf.zeros([self.batch_size, self.img_width, self.img_width, 1], tf.float32)
        # tf.placeholder(tf.float32, [None, None, None])

        self.rnn_steps = []
        with tf.name_scope('RNN'):
            for i in range(seq_length):
                self.rnn_steps.append(
                    self.create_alpha_palette_step(
                        self.fmap, global_mask, reuse=(len(self.rnn_steps) > 0),
                        palette_graph_arch=palette_graph_arch, palette_opts=palette_opts,
                        palette_layer_specs=palette_layer_specs,
                        fmap_filters=fmap_filters))

                global_mask = self.rnn_steps[-1]['global_mask']

        # Get alphas after RNN executes
        alphas = tf.concat([x['alpha'] for x in self.rnn_steps], axis=3)
        self.out_labels = tf.nn.softmax(alphas)
        self.unnorm_alphas = alphas
        # Might need unnormalized version
        # NOTE: Check if these are actually unnormalized
        self._init_out_labels_viz()

        with tf.name_scope('combined_palette'):
            self.palette_colors = tf.concat(
                [tf.expand_dims(x['palette'].patch_colors, axis=1) for x in self.rnn_steps], axis=1)


    def init_goodness_loss(self, is_soft):
        raise RuntimeError('Not implemented')
        # TODO: does this even make sense for total loss set up? Come up with a way that does make sense
        # goodness,gdeb = losses.compute_alpha_prediction_goodness(self.x, alpha, global_mask, step_palette.colors)
        # step_reconstruction_loss = -tf.reduce_sum(goodness) / tf.to_float(self.batch_size)


    def create_alpha_palette_step(self, feature_map, global_mask, reuse, palette_graph_arch, palette_opts,
                                  palette_layer_specs, fmap_filters):
        '''

        :param feature_map: [ Batchsize, width, height, nchannels ]
        :param global_mask: [ Batchsize, width, height ]
        :return:
        '''
        with tf.name_scope('predict_step'):
            # Step1: multiply feature_map and global_mask
            LOG.debug('Feature map %s' % str(feature_map.shape))
            LOG.debug('Global mask %s' % str(global_mask.shape))
            fmap = tf.multiply(feature_map, 1.0 - global_mask)

            # Step2: produce a single alpha
            output = tf.layers.conv2d(fmap, fmap_filters / 2, 1, padding='same', name='alpha_conv1', reuse=reuse)
            output = tf.nn.relu(output)
            output = tf.layers.conv2d(output, 1, 1, padding='same', name='alpha_conv2', reuse=reuse)
            alpha = tf.tanh(output) * 0.5 + 0.5
            LOG.debug('Alpha %s' % str(alpha.shape))

            # Step3: compute weighted histogram, reusing previous image quantization result
            hist, flat_hist = color_ops.hist_count(self.hist_count_info[0], self.hist_count_info[1],
                                                   weights=tf.reshape(alpha, [self.batch_size, -1, 1]),
                                                   normalize=True, squeeze=True)

            # Step4: produce a simple palette encoding
            step_palette, palette_encoding, _, _ = \
                self._init_histogram_graphs(palette_graph_arch,
                                            flat_hist if palette_graph_arch == 0 else hist,
                                            palette_opts, palette_layer_specs, reuse=reuse)

            # Step8: update global mask
            new_global_mask = tf.minimum(1.0, tf.add(global_mask, alpha))
            LOG.debug('New global mask %s' % str(new_global_mask.shape))

            return { 'alpha' : alpha,
                     'input_hist' : flat_hist,
                     'palette' : step_palette,
                     'global_mask' : new_global_mask,
                     'input_global_mask' : global_mask }


    def init_vgg_unet(self, filter_size=4, output_filters=64):
        # Step1: create vgg net:
        if self.img_width != 224:
            raise RuntimeError('VGG pretrained net can only accept 224x224 images')

        with tf.name_scope('vgg_pretrained'):
            self.vgg = vgg16.Vgg16()
            self.vgg.build(self.x_mask_input)

        # Step2: add unet layers, starting at pool5
        # TODO: what filter sizes to use? Should BN be anywhere?
        layer_specs = [
            #(self.vgg.pool5, output_filters * 8, 0.5, False),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.vgg.conv5_3, output_filters * 4, 0.5, False),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.vgg.conv4_3, output_filters * 4, 0.5, False),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.vgg.conv3_3, output_filters * 2, 0.0, False),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.vgg.conv2_2, output_filters , 0.0, False)  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            #(self.vgg.conv1_2, output_filters, 0.0, False)       # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        ]

        with tf.name_scope('unet_decoder'):
            layers = self.init_unet_decoder(layer_specs, filter_size=filter_size)
            output = tf.tanh(layers[-1])
        return output


    # Port of pix2pix tf implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    def init_vanilla_unet(self, ngf=64, output_filters=4, filter_size=4, dropout=None):
        if self.img_width is None:
            raise RuntimeError('Only enabled for image input')

        input = self.x_mask_input

        if dropout is not None:
            input_bw = tf.image.rgb_to_grayscale(input, name='grayscaler')
            if dropout < 1:
                LOG.debug('MASK graph has 4 channel input. RGB dropout is:%.2f',dropout)
                noise_shape = tf.shape(input_bw)
                input_dropped = tf.nn.dropout(input, keep_prob=1-dropout, noise_shape=noise_shape)
                tf.summary.image('Mask_Graph_GrayInput', input_bw, collections=['train'])
                tf.summary.image('Mask_Graph_RGBInput', input_dropped, collections=['train'])

                input = tf.concat([input_bw, input_dropped], axis=-1)
            else:
                LOG.debug('MASK graph has 1 channel input')
                # If dropout is >1 then use only bw image
                input = input_bw


        LOG.info('Encoder input: %s' % str(input))
        layers = []

        with tf.name_scope('vanilla_unet_encoder'):
            # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope("encoder_1"):
                output = tf.layers.conv2d(input, ngf, filter_size, strides=2, padding='same')
                LOG.info('Conv 1: %s' % str(output.shape))
                layers.append(output)

            layer_specs = [
                ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]

            for out_channels in layer_specs:
                idx = len(layers) + 1
                with tf.variable_scope("encoder_%d" % idx):
                    rectified = tf.nn.leaky_relu(layers[-1], 0.2)
                    LOG.info('Leaky relu %d: %s' % (idx, str(rectified.shape)))
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = tf.layers.conv2d(rectified, out_channels, filter_size, strides=2, padding='same')
                    LOG.info('Conv %d: %s' % (idx, str(convolved.shape)))
                    # conv(rectified, out_channels, stride=2)
                    output = tf.keras.layers.BatchNormalization()(convolved)
                    LOG.info('BN %d: %s' % (idx, str(output.shape)))
                    # batchnorm(convolved)
                    layers.append(output)

        layer_specs = [
            (layers[-1], ngf * 8, 0.5, True),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (layers[-2], ngf * 8, 0.5, True),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (layers[-3], ngf * 8, 0.5, True),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (layers[-4], ngf * 8, 0.0, True),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (layers[-5], ngf * 4, 0.0, True),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (layers[-6], ngf * 2, 0.0, True),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (layers[-7], ngf, 0.0, True),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            (layers[-8], output_filters, 0.0, False)
        ]

        with tf.name_scope('unet_decoder'):
            layers.extend(self.init_unet_decoder(layer_specs, filter_size=filter_size))
            output = tf.tanh(layers[-1])
            layers.append(output)
        return output


    def init_oneshot_masks(self,
        npalettes,
        palette_graph_arch,
        palette_opts,
        palette_layer_specs,
        mask_softmax_temp):

        # To get consistency with RNN mask, final unnormalized alphas are called self.alphas
        # NOTE: Haven't checked RNN mask part
        self.unnorm_alphas = self.fmap * mask_softmax_temp

        # Next we apply softmax
        self.out_labels = tf.nn.softmax(self.unnorm_alphas)

        self._init_out_labels_viz()

        # Note: to experiment with onehot again
        # Reshape to [Nbatches x ncolors x nalphas]
        # hist_weights = tf.reshape(self.out_labels, [self.batch_size, -1, tf.shape(self.out_labels)[3]])
        # hist_weights = color_ops.make_onehot(hist_weights)
        # alpha = tf.slice(hist_weights, [0,0,p], [self.batch_size, self.img_width * self.img_width, 1])

        # Next we predict palettes
        self.rnn_steps = []
        for p in range(npalettes):
            alpha = tf.slice(self.out_labels, [0,0,0,p], [self.batch_size, self.img_width, self.img_width, 1])
            hist, flat_hist = color_ops.hist_count(
                self.hist_count_info[0], self.hist_count_info[1],
                weights=tf.reshape(alpha, [self.batch_size, -1, 1]),
                normalize=True, squeeze=True)

            step_palette, palette_encoding, _, _ = \
                self._init_histogram_graphs(palette_graph_arch,
                                            flat_hist if palette_graph_arch == 0 else hist,
                                            palette_opts, palette_layer_specs, reuse=(len(self.rnn_steps) > 0))

            self.rnn_steps.append(
                {'palette' : step_palette,
                 'alpha': alpha,
                 'input_global_mask': tf.zeros([self.batch_size, self.img_width, self.img_width, 1], tf.float32)})

            with tf.name_scope('combined_palette'):
                self.palette_colors = tf.concat(
                    [tf.expand_dims(x['palette'].patch_colors, axis=1) for x in self.rnn_steps], axis=1)


    def _init_out_labels_viz(self):
        self.out_labels_flat = tf.reshape(
            self.out_labels, [self.batch_size, -1, tf.shape(self.out_labels)[-1]])
        perm = [0, 1, 3, 2]
        self.out_labels_viz = tf.tile(
            tf.expand_dims(tf.reshape(tf.transpose(self.out_labels, perm=perm),
                                      [self.batch_size, self.img_width, -1]), axis=3), [1, 1, 1, 3])


    def init_unet_decoder(self, layer_specs, filter_size):
        '''
        Initializes U-Net style decoder, given a list of layer specs, where each is a tuple:
        (skip layer, output_channels, dropuout fraction, use batch_norm?); the first tuple's skip layer is the input to the U-Net.
        :param layer_specs:
        :return:
        '''
        layers = []
        for decoder_layer, (skip_layer, out_channels, dropout, use_bn) in enumerate(layer_specs):
            idx = decoder_layer
            with tf.variable_scope("UNETdecoder_%d" % idx):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = skip_layer
                else:
                    input = tf.concat([layers[-1], skip_layer], axis=3)
                LOG.info('(Decoder input %d: %s)' % (idx, str(input.shape)))

                rectified = tf.nn.relu(input)
                LOG.info('Decoder relu %d: %s' % (idx, str(rectified.shape)))

                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = tf.layers.conv2d_transpose(rectified, filters=out_channels, kernel_size=filter_size, strides=2, padding="same")
                LOG.info('Decoder Conv %d: %s' % (idx, str(output.shape)))

                if use_bn:
                    output = tf.keras.layers.BatchNormalization()(output)
                    LOG.info('Decoder BN %d: %s' % (idx, str(output.shape)))

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                    LOG.info('Decoder dropout %d: %s' % (idx, str(output.shape)))
                layers.append(output)

        return layers

    def init_alpha_binary_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_binary_loss'):
            # Next we pass all the weights through a sigmoid to push large values up and small values down
            sq_labels = tf.reciprocal(1.0 + tf.exp(-20.0 * (self.out_labels - 0.5)))

            # We want the sum of these to be as large as possible
            # The maximum this can be is W * H * 1.0
            sq_labels_sum = tf.reduce_sum(sq_labels) / tf.to_float(self.batch_size)

            sq_labels_loss = self.img_width * self.img_width * 1.0 - sq_labels_sum
            return sq_labels_loss


    def init_alpha_neg_max_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_neg_max_loss'):
            neg_max = -tf.reduce_sum(tf.reduce_max(self.out_labels, axis=3))
            return neg_max / tf.to_float(self.batch_size)


    def init_alpha_binary_v2_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_binary_v2_loss'):
            # Next we pass all the weights through a sigmoid to push large values up and small values down
            sq_labels = self.out_labels ** 2

            # We want the sum of these to be as large as possible
            # The maximum this can be is 1.0
            sq_labels_sum = tf.reduce_mean(sq_labels)
            sq_labels_loss = 1.0 - sq_labels_sum

            return sq_labels_loss

    def init_alpha_entropy_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_entropy_loss'):
            shape = tf.stack([-1, tf.shape(self.out_labels)[-1]])
            out_labels = tf.reshape(self.out_labels, shape)
            unnorm_alphas = tf.reshape(self.unnorm_alphas, shape)

            # cross entropy with self as a numerically stable way to get entropy
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=out_labels,
                logits=unnorm_alphas
            )

            entropy = tf.reduce_mean(entropy)

            return entropy


    def init_alpha_maxmindiff_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_maxmindiff_loss'):
            max_out = tf.reduce_max(self.out_labels, axis=-1)
            min_out = tf.reduce_min(self.out_labels, axis=-1)

            # We want to maximize this value
            # The maximum this can be is 1.0
            loss = 1 - tf.reduce_mean(max_out - min_out)

            return loss

    def init_alpha_tv_loss(self):
        if self.out_labels is None:
            raise RuntimeError('Alpha binary loss cannot be created without out_labels')

        with tf.name_scope('alpha_tv_loss'):
            tv = tf.reduce_sum(tf.image.total_variation(self.out_labels)) / tf.to_float(self.batch_size)
            return tv / tf.to_float(tf.shape(self.out_labels)[-1])

    def init_l2rgb_loss(self, blend):
        if self.restored_image is not None:
            raise RuntimeError('More than one reconstruction loss type is disallowed')

        with tf.name_scope('l2rgb_loss'):
            approx, blend_norm, mapping = color_ops.get_alpha_mask_recon_error(
                self.img_colors, self.palette_colors, self.out_labels_flat, squared=True, blend=blend)
            self.restored_image = tf.reshape(approx, tf.shape(self.x))
            self.mapping = mapping
            return tf.reduce_sum(blend_norm) / tf.to_float(self.batch_size)


    def init_percent_reconstruction_loss(self):
        if self.palette is None:
            raise RuntimeError('Cannot compute percent reconstruction loss without a palette; not implemented')

        with tf.name_scope('percent_loss'):
            delta = 10.0
            self.ok_mask, self.lab_image_error, loss = losses.compute_frac_rep_colors(
                self.x, tf.squeeze(self.palette_colors, axis=1), delta)
            return tf.reduce_sum(loss) / tf.to_float(self.batch_size)


    def init_l2lab_loss(self):
        if self.restored_image is not None:
            raise RuntimeError('More than one reconstruction loss type is disallowed')

        if self.lab_palette_colors is None:
            self.lab_palette_colors = color_ops.rgb2lab_anyshape(self.palette_colors)

        with tf.name_scope('l2lab_loss'):
            approx, blend_norm, mapping = color_ops.get_alpha_mask_recon_error(
                self.lab_img_colors, self.lab_palette_colors, self.out_labels_flat, squared=True, blend=True)
            approx = color_ops.lab2rgb(approx)
            self.restored_image = tf.reshape(approx, tf.shape(self.x))
            self.mapping = mapping
            return tf.reduce_sum(blend_norm) / tf.to_float(self.batch_size)


    def init_palette_mask_loss(self, squashed_reg=0.1, tv_reg=0.05,
                               use_squashed_labels=True, use_external_colors=None):
        '''

        :param squashed_reg:
        :param tv_reg:
        :param use_squashed_labels:
        :param use_external_colors: [Ncolors x 3]
        :return:
        '''
        if self.out_labels is None:
            raise RuntimeError('Must call init_image_to_image encoder graph first')

        with tf.name_scope('mask_loss'):
            # Add total variation to the mask output


            # We normalize squashed labels to compute actual averages
            if use_squashed_labels:
                sq_labels = tf.reciprocal(1.0 + tf.exp(-20.0 * (self.out_labels - 0.5)))
                labels = tf.divide(sq_labels, tf.expand_dims(tf.reduce_sum(sq_labels, axis=3), axis=3))
            else:
                labels = self.out_labels


            # labels: [Nbatches x W x H x npalettes]

            # Loss is:
            # Sum_x,y color(x,y) - alpha-blended-color-of-alpha-regions

            # Each palette's weight: [ Nbatches x npalettes ]
            psum = tf.reduce_sum(tf.reduce_sum(labels, axis=2), axis=1)

            if use_external_colors is not None:
                # [Ncolors x 3] -> [1 x 3 x Ncolors] -> [Nbatches x 3 x Ncolors]
                ave_pcolors = tf.tile(tf.expand_dims(tf.transpose(use_external_colors, perm=[1,0]), axis=0), [self.batch_size, 1, 1])
                LOG.debug('Ave colors %s' % str(ave_pcolors.shape))
            else:
                # Want: [Nbatches x channels x npalettes] average palette colors:
                # 1. [ nbatches x W x H x channels x npalettes ] =
                # [ nbatches x W x H x channels x 1] * [ nbatches x W x H x 1 x npalettes]
                # 2. reduce W, H --> [ nbatches x channels x npalettes]
                # 3. normalize
                ave_pcolors = tf.divide(
                    tf.reduce_sum(
                        tf.reduce_sum(  # [nbatches x W x H x channels] p
                            tf.multiply(tf.expand_dims(self.x, axis=4), tf.expand_dims(self.out_labels, axis=3)), axis=2), axis=1),
                    tf.expand_dims(psum, axis=1))
                # TODO: the part above shuold be replaced with at least a 3-color palette

            # Visualize average colors too
            ave_viz = tf.expand_dims(tf.transpose(ave_pcolors, perm=[0,2,1]), axis=0)
            tf.summary.image('ave_colors_viz', color_ops.to_uint8(ave_viz), collections=['test'])

            # Next for each pixel we get a blend of the ave colors
            # [Nbatches x W x H x channels x npalettes] =
            # [Nbatches x W x H x 1 x npalettes ] * [Nbatches x 1 x 1 x channels x npalettes]
            self.restored_image = tf.reduce_sum(
                tf.multiply(
                    tf.expand_dims(self.out_labels, axis=3),
                    tf.expand_dims(tf.expand_dims(ave_pcolors, axis=1), axis=1)), axis=4)

            # The maximum this can be is W * H * 3
            image_loss = tf.reduce_sum(tf.abs(self.restored_image - self.x)) / tf.to_float(self.batch_size)
            tf.summary.scalar('train_image_loss', image_loss, collections=['train'])
            tf.summary.scalar('test_image_loss', image_loss, collections=['test'])

            self.loss = image_loss + squashed_reg * sq_labels_loss + tv_reg * tv
            tf.summary.scalar('train_loss', self.loss, collections=['train'])
            tf.summary.scalar('test_loss', self.loss, collections=['test'])

            # Add visualization:
            viz = tf.concat([self.x, self.restored_image, self.out_labels_viz], axis=2)
            tf.summary.image('restored_image_viz', color_ops.to_uint8(viz),
                             collections=['test'], max_outputs=20)
        return self.loss


    def init_image_encoder_graph(
            self, palette_opts,
            conv_layers=[(4, 64), (4, 128), (4, 256)],
            fcsizes=[800, 200, 50]):
        self.palette = palette.PaletteHelper(palette_opts)

        current_input = self.x
        LOG.debug('Raw input: %s' % str(current_input.shape))
        for i in range(len(conv_layers)):
            with tf.name_scope("conv%d" % i):
                filter_size = conv_layers[i][0]
                n_filters = conv_layers[i][1]
                conv = tf.layers.conv2d(current_input, n_filters, filter_size, strides=1, padding='same')
                # Or: tf.layers.batch_normalization ?
                LOG.debug('Conv %d: %s' % (i, str(conv.shape)))
                bn = tf.keras.layers.BatchNormalization()(conv)
                LOG.debug('BN %d: %s' % (i, str(bn.shape)))
                act = tf.nn.relu(bn)
                LOG.debug('Act %d: %s' % (i, str(act.shape)))
                pool = tf.layers.max_pooling2d(inputs=act, pool_size=[4, 4], strides=4)
                LOG.debug('Pool %d: %s' % (i, str(pool.shape)))
                current_input = pool

        flat_shape = [tf.shape(current_input)[0], conv_layers[-1][0] ** 2 * conv_layers[-1][1]]
        current_input = tf.reshape(current_input, tf.stack(flat_shape))
        LOG.debug('Input to FC size: %s' % str(current_input))
        layers, self.encoding, palette_flat_colors, self.levels, self.wind = \
            self._add_fc_encoder_graph(current_input, palette_opts, fcsizes)
        LOG.debug('Output of FC size: %s' % str(self.encoding))

        self.palette.init_deterministic_decoder(palette_flat_colors, self.wind)
        self.palette_colors = tf.expand_dims(self.palette.patch_colors, axis=1)
        #raise RuntimeError('Done')


    def init_hist_encoder_graph(self, palette_opts, fcsizes=[700, 200, 50]):
        self._init_histogram_graph_standalone(0, self.flat_hist, palette_opts, graph_opts=fcsizes)


    def init_3d_hist_encoder_graph(self, palette_opts, layer_specs):
        self._init_histogram_graph_standalone(4, self.hist, palette_opts, graph_opts=layer_specs)


    def _init_histogram_graph_standalone(self, graph_arch, input, palette_opts, graph_opts):
        tf.summary.histogram('histogram_histogram', self.flat_hist, collections=['train'])
        self.palette,self.encoding,self.levels,self.wind = self._init_histogram_graphs(
            graph_arch, input, palette_opts, graph_opts, reuse=False)

        self.palette_colors = tf.expand_dims(self.palette.patch_colors, axis=1)
        self.result_vars['vertex_colors'] = self.palette.colors
        self.result_vars['patch_colors'] = self.palette.patch_colors


    def _init_histogram_graphs(self, graph_arch, input, palette_opts, graph_opts, reuse):
        '''
        Graph arch:
        0 - FC graph
        4 - 3D conv graph
        :param graph_arch:
        :param palette_opts:
        :param graph_opts:
        :return:
        '''
        tmp_palette = palette.PaletteHelper(palette_opts)

        # Encoder --------------------------------------------------------------------------------
        with tf.variable_scope('palette_graph'):
            if graph_arch == 0:
                layers, encoding, palette_flat_colors, levels, wind = \
                    self._add_fc_encoder_graph(input, palette_opts, fcsizes=graph_opts, reuse=reuse)
            elif graph_arch == 4:
                layers, encoding, palette_flat_colors, levels, wind = \
                    self._add_3d_hist_encoder(input, palette_opts, reuse=reuse)  # TODO: add layer specs
            else:
                raise RuntimeError('Could not recognize graph architecture %d (see --encoder_mode and --palette_graph_mode' % graph_arch)
        tmp_palette.init_deterministic_decoder(palette_flat_colors, wind)

        return tmp_palette, encoding, levels, wind


    def compute_enc_dims(self, palette_opts):
        def __get_wind_channels(wind):
            if wind > 1:
                wind += 1

            return wind

        return palette_opts.n_colors * palette_opts.n_channels + \
            (self.palette.n_tri if palette_opts.discrete_continuous else 0)+ \
            (__get_wind_channels(palette_opts.wind_nchannels))


    def _add_3d_hist_encoder(self, input, palette_opts,
                             layer_specs=[(3, 64, 2), (3, 128, 2), (3, 256, 3), (1, 64, 1)], reuse=False):
        '''

        :param input: [Nbatches, nbins, nbins, nbins, nchannels]
        :return:
        '''
        if len(input.shape) == 4:
            current_input = tf.expand_dims(input, axis=4)
        else:
            current_input = input
        LOG.info('PG input - %s' % str(current_input.shape))
        # TODO: add dropout?

        enc_dims = self.compute_enc_dims(palette_opts)

        tmp_lspecs = [x for x in layer_specs]
        tmp_lspecs.append((1, enc_dims, 1))
        layers = []
        for idx, (fsize, out_channels, stride) in enumerate(tmp_lspecs):
            with tf.variable_scope("pg_%d" % idx):
                rectified = tf.nn.leaky_relu(current_input, 0.2)
                LOG.info('PG - Leaky relu %d: %s' % (idx, str(rectified.shape)))
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = tf.layers.conv3d(rectified, out_channels, fsize, strides=stride, padding='same', reuse=reuse)
                LOG.info('PG - Conv %d: %s' % (idx, str(convolved.shape)))
                # conv(rectified, out_channels, stride=2)
                output = tf.keras.layers.BatchNormalization()(convolved, reuse=reuse)
                LOG.info('PG - BN %d: %s' % (idx, str(output.shape)))
                # batchnorm(convolved)
                current_input = output
                layers.append(current_input)

        with tf.name_scope("encoding"):
            palette_encoding = tf.squeeze(tf.tanh(current_input) * 0.5 + 0.5)
            flat_colors, levels, wind = self._interpret_palette_encoding(palette_encoding, palette_opts)

        return layers, palette_encoding, flat_colors, levels, wind


    def _add_fc_encoder_graph(self, input, palette_opts, fcsizes, reuse=False):
        # Compute size of last layer, given palette options
        enc_dims = self.compute_enc_dims(palette_opts)

        LOG.info('Setting up graph with encoding size of %d' % enc_dims)
        enc_fcsizes = [x for x in fcsizes]
        enc_fcsizes.append(enc_dims)

        # Add FC layers
        layers = []
        current_input = input
        for i in range(len(enc_fcsizes)):
            scope_i = "fc%d" % i
            with tf.name_scope(scope_i):
                actfn = tf.nn.relu
                if i == len(enc_fcsizes) - 1:
                    actfn = tf.nn.tanh  # Tanh in last layer to clamp to [-1, 1] range for color
                # Uses xavier initialization and relu by default
                current_input = Dense(enc_fcsizes[i], activation=actfn)(current_input)
                                                                  #scope='var' + scope_i, reuse=reuse)
                LOG.debug('PG - FC %d: %s' % (i, str(current_input.shape)))
                layers.append(current_input)
                # TODO: add dropout here

        # Interpret encoding
        with tf.name_scope("encoding"):
            palette_encoding = tf.divide(tf.add(current_input, 1.0), 2.0)
            flat_colors, levels, wind = self._interpret_palette_encoding(palette_encoding, palette_opts)
        return layers, palette_encoding, flat_colors, levels, wind


    def _interpret_palette_encoding(self, palette_encoding, palette_opts):
        LOG.info('Palette encoding %s' % str(palette_encoding))
        flat_colors = tf.slice(
                palette_encoding, [0, 0], [self.batch_size, palette_opts.n_colors * palette_opts.n_channels])

        if palette_opts.discrete_continuous:
            levels = tf.slice(
                palette_encoding, [0, palette_opts.n_colors * palette_opts.n_channels], [self.batch_size, self.palette.n_tri])
            wind_start = palette_opts.n_colors * palette_opts.n_channels + self.palette.n_tri
        else:
            levels = None
            wind_start = palette_opts.n_colors * palette_opts.n_channels

        if palette_opts.wind_nchannels > 0:
            if palette_opts.wind_nchannels == 1:
                wind = tf.slice(palette_encoding, [0, wind_start], [self.batch_size, 1]) * 2.0 - 1.0  # -1..1 range
            elif palette_opts.wind_nchannels == 2:
                wind = tf.nn.softmax(tf.slice(palette_encoding, [0, wind_start], [self.batch_size, 3]))
            elif palette_opts.wind_nchannels == 3:
                wind = tf.concat([
                    tf.slice(palette_encoding, [0, wind_start], [self.batch_size, 1]) * 2.0 - 1.0,
                    tf.nn.softmax(tf.slice(palette_encoding, [0, wind_start + 1], [self.batch_size, 3]))],
                    axis=1)
        else:
            wind = None

        return flat_colors, levels, wind


    # TODO: try Wasserstein GAN loss here
    def init_kl_loss(self):
        if self.flat_hist is None:
            raise RuntimeError('Must call init_histogram_computation for kl_loss')

        with tf.name_scope("loss"):
            if self.palette is not None:
                self.palette.init_histogram(self.n_bins)
                js_loss, self.loss_vars = losses.construct_js_divergence(
                    self.flat_hist, self.palette.flat_hist)
                self.result_vars['palette_flat_hist'] = self.palette.flat_hist
            elif self.rnn_steps is not None:
                sigma = 1.0 / self.n_bins / 2.0
                sigma_sq = sigma * sigma
                self.rnn_palettes_hist, hist_vars = color_ops.compute_rbf_hist(self.palette_colors, self.n_bins, sigma_sq)
                js_loss, self.loss_vars = losses.construct_js_divergence(
                    self.flat_hist, self.rnn_palettes_hist)
                self.result_vars['rnn_palettes_flat_hist'] = self.rnn_palettes_hist
            self.result_vars['flat_hist'] = self.flat_hist

            return js_loss / tf.to_float(self.batch_size)


    def init_onesided_kl_loss(self):
        if self.flat_hist is None:
            raise RuntimeError('Must call init_histogram_computation for kl_loss')

        if self.palette is None:
            raise RuntimeError('No palette present for redundancy kl loss')

        with tf.name_scope("loss"):
            self.palette.init_histogram(self.n_bins)

            # HACK: this assumes we are training on 32x32 patches
            hist_vars = color_ops.compute_patch_hist(
                self.x, hist_subdivs=self.n_bins, patch_width=8, patch_stride=4)
            norm_hist = hist_vars['norm_max_hist']

            # Note: we use max normalization typically, so we need to re-normalize by sum here
            #hist_sum = tf.reduce_sum(self.flat_hist, axis=1)
            #norm_hist = tf.divide(self.flat_hist, tf.expand_dims(hist_sum, axis=1))
            div = losses.construct_kl_divergence(self.palette.flat_hist, norm_hist)

            return div / tf.to_float(self.batch_size)


    def todo_levels_alpha_loss(self, levels_reg_weight=0.1, alpha_reg_weight=1.0):
        if levels_reg_weight > 0.0 and self.levels is not None:
            with tf.name_scope('levels_reg'):
                lreg = levels_reg_weight * tf.reduce_sum(tf.abs(self.levels))
                loss = loss + lreg
                tf.summary.scalar('train_levelreg_loss', lreg, collections=['train'])
                tf.summary.scalar('test_levelreg_loss', lreg, collections=['test'])

        if alpha_reg_weight > 0.0 and self.palette.opts.use_alpha:
            with tf.name_scope('alpha_reg'):
                alphas = self.palette.alphas
                areg = alpha_reg_weight * tf.reduce_sum(tf.abs(alphas))
                tf.summary.scalar('train_alphareg_loss', areg, collections=['train'])
                tf.summary.scalar('test_alphareg_loss', areg, collections=['test'])
                loss = loss + areg

        self.loss = tf.divide(loss, tf.to_float(self.batch_size))  # tf.reduce_sum(tf.image.total_variation(self.z)))
        tf.summary.scalar('train_loss', self.loss, collections=['train'])
        tf.summary.scalar('test_loss', self.loss, collections=['test'])
        return self.loss


    def init_visualization(self, indexes, test_summary_collections, show_reconstruction=True):
        if self.img_width is None:
            return

        if self.palette is not None:
            self.init_palette_visualization(indexes, test_summary_collections,
                                            show_reconstruction=show_reconstruction)

        if self.rnn_steps is not None:
            self.init_rnn_visualization(indexes, test_summary_collections,
                                        show_reconstruction=show_reconstruction)

        if self.out_labels_viz is not None:
            if self.restored_image is not None:
                viz = tf.concat([self.x, self.restored_image, self.out_labels_viz], axis=2)
            else:
                viz = tf.concat([self.x, self.out_labels_viz], axis=2)
            tf.summary.image('restored_image_viz', color_ops.to_uint8(viz),
                             collections=test_summary_collections, max_outputs=10)


    def init_rnn_visualization(self, indexes, test_summary_collections, show_reconstruction=True):
        # Image:
        # image * global_mask; output_alpha; output_alpha * image; palette viz
        def composite_row(idx):
            return (lambda gm, am, al, p:
                    img_util.concat_images([gm[idx], al[idx], am[idx], p]))

        img_gm = []
        img_am = []
        img_al = []
        visualizations = []
        with tf.name_scope('viz'):
            for s in range(len(self.rnn_steps)):
                img_gm.append(self.x * (1.0 - self.rnn_steps[s]['input_global_mask']))
                step_alpha = tf.slice(self.out_labels,
                                      [0,0,0,s],
                                      [self.batch_size, self.img_width, self.img_width, 1])
                img_am.append(self.x * step_alpha)
                img_al.append(tf.tile(step_alpha, [1,1,1,3]))

            for idx in indexes:
                imgs = []
                for s in range(len(self.rnn_steps)):
                    pimg = self.rnn_steps[s]['palette'].get_viz_for_idx_py_func(idx, self.img_width)
                    imgs.append(tf.py_func(composite_row(idx),
                                           [img_gm[s], img_am[s], img_al[s], pimg], tf.float32))
                idx_img = tf.concat(imgs, axis=0)
                visualizations.append(color_ops.to_uint8(tf.expand_dims(idx_img, axis=0)))
                if idx == 0:
                    self.result_viz_vars['alphas'] = visualizations[-1]

            for col in test_summary_collections:
                with tf.name_scope(col + '_viz'):
                    for idx in range(len(visualizations)):
                        tf.summary.image('palette_viz%d' % idx, visualizations[idx], collections=[col])



    def init_palette_visualization(self, indexes, test_summary_collections, show_reconstruction=True):
        '''

        :param indexes: list of test image indexes, for which to visualize paltte
        :return:
        '''
        if self.img_width is None:
            raise RuntimeError('Palette visualization only implemented for image input')

        def composite2_closure(idx):
            return (lambda i, viz0: img_util.concat_images([i[idx], viz0]))

        def composite3_closure(idx):
            return (lambda i, viz0, viz1:img_util.concat_images([i[idx], viz0, viz1]))

        def reconstruct_closure(idx, p):
            return (lambda icolors, pcolors:
                    color_ops.get_numpy_hashed_approximation(icolors[idx], np.squeeze(pcolors[idx]),
                                                             use_source_alpha=p.opts.use_alpha))
        visualizations = []
        with tf.name_scope('viz'):
            prev_approx = None
            for idx in indexes:
                # TODO: figure out why the conditional did not work
                # img = tf.cond(self.batch_size < idx, lambda: tf.py_func(viz, [self.z, self.palette_verts], tf.float32),
                #              lambda: tf.zeros([300,300,3], dtype=tf.float32))
                palette_img = self.palette.get_viz_for_idx_py_func(idx, self.img_width)

                if idx == 0:
                    self.result_viz_vars['palette'] = tf.expand_dims(palette_img, axis=0)
                    self.result_viz_vars['encoding_img'] = self.palette.encoding_img
                    self.result_vars['encoding_img'] = self.palette.encoding_img

                if show_reconstruction:
                    if self.restored_image is not None:
                        recon = self.restored_image[idx,...]
                    else:
                        if prev_approx is not None:
                            with tf.control_dependencies([prev_approx]):
                                approx, metric, best_idx = tf.py_func(reconstruct_closure(idx, self.palette),
                                                                      [self.img_colors, self.palette.patch_colors],
                                                                      [tf.float32, tf.float32, tf.int64])
                        else:
                            approx, metric, best_idx = tf.py_func(reconstruct_closure(idx, self.palette),
                                                                  [self.img_colors, self.palette.patch_colors],
                                                                  [tf.float32, tf.float32, tf.int64])
                        prev_approx = approx
                        recon = tf.reshape(approx, [self.img_width, self.img_width, 3])
                    recon = tf.clip_by_value(recon, 0.0, 1.0)
                    if idx == 0:
                        self.result_viz_vars['recon'] = tf.expand_dims(recon, axis=0)
                    img = tf.py_func(composite3_closure(idx), [self.x, palette_img, recon], tf.float32)
                else:
                    img = tf.py_func(composite2_closure(idx), [self.x, palette_img], tf.float32)

                visualizations.append(color_ops.to_uint8(tf.expand_dims(img, axis=0)))

            for col in test_summary_collections:
                with tf.name_scope(col + '_viz'):
                    for idx in range(len(visualizations)):
                        tf.summary.image('palette_viz%d' % idx, visualizations[idx], collections=[col])



    def encoding_to_images(self, colors):
        '''

        :param colors:
        :return:
        '''
        imgs = []
        for i in range(colors.shape[0]):
            imgs.append(colors[i, :, :].reshape([1,colors.shape[1],3]))
        return imgs
