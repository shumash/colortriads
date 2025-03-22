import tensorflow as tf
from learn.tf.log import LOG

class Discriminator(object):
    '''
    Network for discriminating between true images and edited images.
    Given image answers the question: has this image been edited? 1 - yes (edited), 0 - no (original, real image)

    Modeling options:
    D(image) -> original or color edited?
    D(image, image) -> which is fake?


    Architectures:
    1. Image encoder (VGG like) -> e^x / (1 + e^x)
    2. Patch based architecture (maybe wrong thing, as we may need context to discriminate edits properly)
    3. Coupled 2-image architecture:

    img0 -> | | | | -> \
            siamese      -> concat features -> conv, fc stuff -> softmax  (note: loss is identical to regular one)
    img1 -> | | | | -> /


    Other options:
    - allow importing pre-trained VGG

    Detailed architecture:
    - Vanilla: DCGAN architecture
    - Vanilla: VGG architecture
    - Better:
      > conceptually want to detect pixelation artifacts; uneven shading of previously evenly shaded areas.
    '''

    def __init__(self, coupled=False):
        self.strid = 'DISC'
        self.coupled = coupled
        self.img_width = None
        self.x = None
        self.batch_size = None
        self.y = None
        self.raw_y = None
        self.loss = None
        self.loss_elements = None
        self.gan_loss_elements = None


    def init_image_input(self, img_width):
        if self.x is not None:
            raise RuntimeError('Discriminator input can only be initialized once')

        self.img_width = img_width

        with tf.name_scope(self.strid):
            with tf.name_scope("input"):
                size = [None, img_width, img_width, 3]
                if self.coupled:
                    size.append(2)
                self.x = tf.placeholder(tf.float32, size, name="disc_input")
                self.batch_size = tf.shape(self.x)[0]


    def init_computed_image_input(self, img_width, img_tensor):
        if self.x is not None:
            raise RuntimeError('Discriminator input can only be initialized once')

        self.img_width = img_width
        self.x = img_tensor
        self.batch_size = tf.shape(self.x)[0]


    def init_vanilla_encoder(self):
        pass


    def init_vgg_encoder(self):
        pass


    def init_coupled_architecture(self):
        pass


    def init_discriminator_loss(self, y_truth):
        if self.y is None:
            raise RuntimeError('Must initialized graph first')

        # Should be softmax, not sigmoid
        self.loss_elements = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_y, labels=y_truth)
        self.loss = tf.reduce_sum(self.loss_elements) / tf.to_float(self.batch_size)
        return self.loss_elements, self.loss


    def init_gan_loss_elements(self):
        if self.y is None:
            raise RuntimeError('Must initialized graph first')

        self.gan_loss_elements = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_y, labels=tf.zeros_like(self.raw_y))
        return self.gan_loss_elements


    # Reference: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
    # df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
    # nf1: number of filters in 1st layer
    def init_dcgan_graph(self, nf1=64, filter_size=5, reuse=False):
        if self.x is None:
            raise RuntimeError('Must initialize input first')

        LOG.info('Initializing DCGAN graph with input of shape %s' % str(tf.shape(self.x)))
        print('Initializing DCGAN graph with input of shape %s' % str(tf.shape(self.x)))
        with tf.name_scope(self.strid):  # Groups graph operations in the viz
            with tf.variable_scope(self.strid) as scope:
                if reuse:
                    scope.reuse_variables()

                layer_specs = [ nf1, nf1 * 2, nf1 * 4, nf1 * 8 ]
                layers = [ self.x ]

                idx = 0
                for out_channels in layer_specs:
                    idx += 1

                    convolved = tf.layers.conv2d(
                        layers[-1], out_channels, filter_size, strides=2, padding='same',
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                    layers.append(convolved)
                    LOG.info('DISC Conv %d: %s' % (idx, str(layers[-1].shape)))

                    if idx > 1:
                        normalized = tf.contrib.layers.batch_norm(convolved)
                        layers.append(normalized)
                        LOG.info('DISC BatchNorm %d: %s' % (idx, str(layers[-1].shape)))

                    rectified = tf.nn.leaky_relu(layers[-1], 0.2)
                    layers.append(rectified)
                    LOG.info('DISC LeakyRelu %d: %s' % (idx, str(layers[-1].shape)))

                oshape = tf.shape(layers[-1])
                layers.append(tf.reshape(layers[-1], [-1, oshape[1] * oshape[2] * oshape[3]]))
                LOG.info('DISC Reshaped: %s' % str(layers[-1].shape))

                # Now apply simple matrix multiplication (this makes it not fully convolutional; ugly)
                osize = layer_specs[-1] * (self.img_width / (2 ** len(layer_specs))) ** 2
                matrix = tf.get_variable("matrix", [osize, 1], tf.float32,
                                         initializer=tf.random_normal_initializer(stddev=0.02))
                bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))
                self.raw_y = tf.matmul(layers[-1], matrix) + bias

                self.y = tf.sigmoid(self.raw_y)
                LOG.info('DISC output: %s' % str(self.y.shape))
