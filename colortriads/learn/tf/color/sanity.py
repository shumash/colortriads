import util.test_util as test_util
import learn.tf.color.encoder as colorenc
import learn.tf.color.color_ops as color_ops
import util.hist
import util.io
import scipy.misc
import tensorflow as tf

def load_histograms():
    hists = {}
    images = {}
    for s in set(["red_blue", "red_blue_grad", "red_yellow",
                  "red_yellow_grad", "red_yellow_grad_blue_acc"]):
        h = colorenc.read_hist_to_row(
            test_util.getAbsoluteTestdataPath(['learn', 'tf', 'hist', s + '.png.hist']))
        img = util.io.read_float_img(
            test_util.getAbsoluteTestdataPath(['learn', 'tf', 'hist', s + '.png']))
        hists[s] = h
        images[s] = img
    return (images, hists)

def img2img_loss_sanity(images, n_bins):
    tf.reset_default_graph()

    # image
    img_width = images.items()[0][1].shape[1]
    z = tf.placeholder(tf.float32, [None] + [i for i in images.items()[0][1].shape])

    # image for histogram
    X = tf.placeholder(tf.float32, [None] + [i for i in images.items()[0][1].shape])
    image_colors = tf.reshape(X, [-1, img_width * img_width, 3])
    hist, hist_flat, deb = color_ops.compute_hist(image_colors, n_bins)
    hist_sum = tf.reduce_sum(hist_flat, axis=2)
    norm_hist = tf.divide(tf.squeeze(hist_flat, axis=1), hist_sum)

    # indexer that helps compute the loss
    raw_indexer, indexer = colorenc.create_hist_indexer(n_bins)
    # standard deviation
    sigma = 1.0 / n_bins / 2.0
    sigma_sq = sigma * sigma

    # construct loss from all this
    loss_vars = colorenc.ColorAutoencoder.construct_kl_loss(
        norm_hist, z, indexer, sigma_sq, eps=1.0e-29)
    loss_vars['x'] = X
    loss_vars['z'] = z
    loss_vars['raw_indexer'] = raw_indexer
    loss_vars['indexer'] = indexer
    loss_vars['sigma'] = sigma
    return loss_vars


def loss_sanity(images, hists, n_bins):
    tf.reset_default_graph()

    # histogram
    X = tf.placeholder(tf.float32, [None, n_bins * n_bins * n_bins])
    # image
    z = tf.placeholder(tf.float32, [None] + [i for i in images.items()[0][1].shape])
    # indexer that helps compute the loss
    raw_indexer,indexer = colorenc.create_hist_indexer(n_bins)
    # standard deviation
    sigma = 1.0 / n_bins / 2.0
    sigma_sq = sigma * sigma

    # construct loss from all this
    loss_vars = colorenc.ColorAutoencoder.construct_kl_loss(
        X, z, indexer, sigma_sq, eps=1.0e-29)
    loss_vars['x'] = X
    loss_vars['z'] = z
    loss_vars['raw_indexer'] = raw_indexer
    loss_vars['indexer'] = indexer
    loss_vars['sigma'] = sigma
    return loss_vars
