import tensorflow as tf
import learn.tf.color.color_ops as color_ops

def construct_js_divergence(hist0, hist1, min_hist_bin_val=1.0e-29):
    '''
    Computes Jensen-Shannon divergence, a symmetrization of KL divergence;
    i.e. constructs a kl loss subgraph and outputs all main vars in a dict.

    Inputs:
    :param hist:
    :param image:
    :param indexer:
    :param sigma:
    :return:
    '''
    # Step1: bin colors from the output
    #res = color_ops.bin_colors(image, indexer, sigma_sq)

    # Step2: simply compute KL divergence, i.e. D(P||Q) + D(Q||P) =
    # Sum( P(i) log (P(i)/Q(i)) + Q(i) log (Q(i)/P(i)) ) =
    # p log p - p log q + q log q - q log p = (p - q) ( log p - log q )

    #res['bins_sum'] = tf.reduce_sum(res['bins'], axis=1)
    #res['bins_norm'] = tf.divide(res['bins'], tf.reshape(res['bins_sum'], [-1, 1]))

    with tf.name_scope("js_div"):
        res = dict()
        res['q'] = tf.maximum(min_hist_bin_val, hist0)
        res['p'] = tf.maximum(min_hist_bin_val, hist1)
        res['pq_log_diff'] = tf.subtract(tf.log(res['p']), tf.log(res['q']))
        res['pq_diff'] = tf.subtract(res['p'], res['q'])
        res['div_elem'] = tf.multiply(res['pq_diff'], res['pq_log_diff'])
        divergence = tf.reduce_sum(res['div_elem'], name='Loss')

    return divergence, res


def construct_kl_divergence(hist0, hist1, min_hist_bin_val=1.0e-29):
    '''
    computes KL(P||Q) = -sum P(i) * (log(Q(i) - log(P(i)))
    :param hist0:
    :param hist1:
    :param min_hist_bin_val:
    :return:
    '''
    p = tf.maximum(min_hist_bin_val, hist0)
    q = tf.maximum(min_hist_bin_val, hist1)
    qp_lg = tf.subtract(tf.log(q), tf.log(p))
    elems = p * qp_lg
    res = -1.0 * tf.reduce_sum(elems, name="KL")
    return res


def squash_lab_dist_to_goodness(x, tight=False):
    '''
    Squashes lab errors (L2 distances) to the range [1, -1], where values < ~7 are "good" (positive) and values > 7 gradually
    approach -1. Lab error of 0 attains maximum goodness of 1.

    :param x: tensor of any shape
    :return: same shape tensor squashed
    '''
    if tight:
        return 1.0 - 2.0 * tf.reciprocal(1.0 + tf.exp(-1.0 * (x-6.0)) + tf.exp(-0.1 * (x-6.0)))
    else:
        return 1 - 2.0 * tf.reciprocal(1.0 + tf.exp(-0.05 * (x-50.0)))
    #return apply_generalized_logistic_function(x, A=-1.0, K=1.0, B=-1.5, v=19.0, Q=0.5, shift=-6.0)


def compute_alpha_prediction_goodness(image, new_alpha, total_mask, palette_colors, tight=False):
    '''

    :param image: [Nbatches, width, height, 3]
    :param new_alpha: [Nbatches, width, height]
    :param total_mask: [Nbatches, width, height]
    :param palette_colors: [Nbatches, Ncolors, nchannels]
    :return:
    '''
    nbatches = tf.shape(image)[0]
    image_colors = tf.reshape(image, [nbatches, -1, 3])
    recon,lab_image_error = color_ops.get_best_color_recon_error(image_colors, palette_colors, use_lab=True)

    color_goodness = squash_lab_dist_to_goodness(lab_image_error)
    delta = tf.reshape(tf.maximum(0.0, new_alpha - total_mask), [nbatches, -1])

    goodness = tf.reduce_sum(tf.multiply(color_goodness, delta), axis=1)
    weighted_error = tf.reduce_sum(tf.multiply(lab_image_error, delta), axis=1)

    debug = {'recon' : tf.reshape(recon, tf.shape(image)),
             'lab_error': tf.reshape(lab_image_error, [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]]),
             'goodness': tf.reshape(color_goodness, [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]]),
             'weighted_error': weighted_error
             }
    return goodness, debug #goodness, debug


def compute_frac_rep_colors(orig, palette_colors, delta):
    '''

    :param orig: [Nbatches x width x width x 3]
    :param reconstructed:  [ Nbatches x width x width x 3 ]
    :param delta: delta in lab
    :return:
    '''
    nbatches = tf.shape(orig)[0]
    orig_colors = tf.reshape(orig, [nbatches, -1, 3])
    recon, lab_image_error = color_ops.get_best_color_recon_error(orig_colors, palette_colors, use_lab=True)

    total_pixels = tf.shape(orig_colors)[1]

    ok_mask = tf.cast(tf.less_equal(lab_image_error, delta), dtype=tf.float32)
    res = tf.reduce_sum(ok_mask)

    return ok_mask, lab_image_error, res / tf.to_float(total_pixels)

