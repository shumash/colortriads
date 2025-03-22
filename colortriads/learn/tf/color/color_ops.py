import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np

import learn.tf.util as tf_util
from learn.tf.log import LOG

# UTILITIES ------------------------------------------------------------------------------------------------------------
def create_image(w, h, colors):
    '''
    Creates image of w x h, with colors = [(r,g,b), (r,g,b)...]
    '''
    nchannels = len(colors[0])
    im = np.zeros([h,w,nchannels], dtype=np.float32)
    for x in range(w):
        for y in range(h):
            idx = y * w + x
            for i in range(nchannels):
                im[y,x,i] = colors[idx][i]
    return im

def to_uint8(img):
    out = tf.cast(img * 255.0, tf.uint8)
    return out


# CONVERSIONS ----------------------------------------------------------------------------------------------------------
# Code is a debugged version of: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

# SRGB <--> Linear RGB -----------------------------
def srgb2linear_rgb(srgb_pixels):
    '''
    :param srgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with linearized RGB
    '''
    with tf.name_scope("srgb2linrgb"):
        linear_mask = tf.stop_gradient(tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32))
        exponential_mask = tf.stop_gradient(tf.cast(srgb_pixels > 0.04045, dtype=tf.float32))
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    return rgb_pixels


def linear_rgb2srgb(rgb_pixels):
    '''
    :param rgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with RGB (for SRGB)
    '''
    with tf.name_scope("linrgb2srgb"):
        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
        exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask
    return srgb_pixels


# Linear RGB <--> XYZ -----------------------------
def linear_rgb2xyz(rgb_pixels):
    '''
    :param rgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with XYZ
    '''
    with tf.name_scope("linrgb2xyz"):
        rgb_to_xyz = tf.constant([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ])
        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
    return xyz_pixels


def xyz2linear_rgb(xyz_pixels):
    '''
    :param xyz_pixels: has shape [N, 3] with X,Y,Z in each row
    :return: same shape with RGB
    '''
    with tf.name_scope("xyz2linrgb"):
        xyz_to_rgb = tf.constant([
            #     r           g          b
            [3.2404542, -0.9692660, 0.0556434],  # x
            [-1.5371385, 1.8760108, -0.2040259],  # y
            [-0.4985314, 0.0415560, 1.0572252],  # z
        ])
        rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
        # avoid a slightly negative number messing up the conversion
        rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
    return rgb_pixels


# XYZ <--> LAB -------------------------------------
def xyz2lab(xyz_pixels):
    '''
    :param xyz_pixels: has shape [N, 3] with X,Y,Z in each row
    :return: same shape with LAB
    '''
    with tf.name_scope("xyz2lab"):
        Xn = 0.95047
        Yn = 1.000
        Zn = 1.08883
        delta = 6.0 / 29.0
        D3 = delta ** 3.0
        D2INV3 = 1.0 / (3 * (delta ** 2))

        xyz_normalized_pixels = tf.multiply(xyz_pixels, [1.0 / Xn, 1.0 / Yn, 1.0 / Zn])

        linear_mask = tf.stop_gradient(tf.cast(xyz_normalized_pixels < D3, dtype=tf.float32))
        exponential_mask = tf.stop_gradient(tf.cast(xyz_normalized_pixels >= D3, dtype=tf.float32))

        eps=1.0e-8  # stabilize cubed root gradient
        fxfyfz_pixels = (xyz_normalized_pixels * D2INV3 + 4.0 / 29) * linear_mask + \
                        (tf.pow(xyz_normalized_pixels+eps, (1.0 / 3.0))) * exponential_mask

        # convert to lab
        fxfyfz_to_lab = tf.constant([
            #  l       a       b
            [0.0, 500.0, 0.0],  # fx
            [116.0, -500.0, 200.0],  # fy
            [0.0, 0.0, -200.0],  # fz
        ])
        lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
    return lab_pixels


def lab2xyz(lab_pixels):
    '''
    :param lab_pixels: has shape [N, 3] with L,A,B in each row
    :return: same shape with XYZ
    '''
    with tf.name_scope("lab2xyz"):
        # convert to fxfyfz
        lab_to_fxfyfz = tf.constant([
            #   fx      fy        fz
            [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
            [1 / 500.0, 0.0, 0.0],  # a
            [0.0, 0.0, -1 / 200.0],  # b
        ])
        fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

        # convert to xyz
        epsilon = 6 / 29.0
        linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
        exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
        xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + \
                     (fxfyfz_pixels ** 3) * exponential_mask
        # denormalize for D65 white point
        xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])
    return xyz_pixels


# Top level converters: SRGB <--> LAB ----------------------
def rgb2lab(srgb_pixels):
    '''
    Converts SRGB colors to CieLAB.

    :param srgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: tensor of shape [N, 3], with L,A,B in each row
    '''
    with tf.name_scope("rgb2lab"):
        rgb_pixels = srgb2linear_rgb(srgb_pixels)
        xyz_pixels = linear_rgb2xyz(rgb_pixels)
        lab_pixels = xyz2lab(xyz_pixels)
    return lab_pixels


def lab2rgb(lab_pixels):
    '''
    Converts CieLAB colors to SRGB.

    :param lab_pixels: has shape [N, 3], with L,A,B in each row
    :return: tensor of shape [N, 3] with R,G,B in each row
    '''
    with tf.name_scope("lab2rgb"):
        xyz_pixels = lab2xyz(lab_pixels)
        rgb_pixels = xyz2linear_rgb(xyz_pixels)
        srgb_pixels = linear_rgb2srgb(rgb_pixels)
    return srgb_pixels


def rgb2lab_anyshape(colors):
    '''

    :param colors: [any shape ... , 3 or more channels]
    :return:
    '''
    return tf.reshape(rgb2lab(tf.reshape(slice_rgb(colors), [-1, 3])), tf.shape(colors))


# PATCHES --------------------------------------------------------------------------------------------------------------
def get_patches(images, pwidth, stride=-1):
    '''
    Extracts patches from a set of images, by default uses stride s.t. patches do not overlap.

    :param images: has shape [nbatches, height, width, Nchannels].
    :return: (patches, patch_colors, patch_images), where
    patches - has shape [ nbatches, row-coord, col-coord, pwidth x pwidth x Nchannels ]
    patch_colors - has shape [ nbatches, row-coord x col-coord, pwidth x pwidth, Nchannels ]
    patch_images - has shape [ nbatches, row-coord x col-coord, pwidth, pwidth, NChannels ]
    '''
    if stride < 0:
        stride = pwidth

    with tf.name_scope("get_patches"):
        nchannels = tf.shape(images)[-1]
        patches = tf.extract_image_patches(
            images, ksizes=[1, pwidth, pwidth, 1], strides=[1, stride, stride, 1], rates=[1,1,1,1], padding='VALID')
        patch_colors = tf.reshape(patches, [tf.shape(patches)[0], -1, tf.cast(tf.shape(patches)[-1]/nchannels, tf.int32), nchannels ])
        patch_images = tf.reshape(patch_colors, [tf.shape(patch_colors)[0], tf.shape(patch_colors)[1], pwidth, pwidth, nchannels])
    return (patches, patch_colors, patch_images)


# HISTOGRAMS -----------------------------------------------------------------------------------------------------------

def compute_hist(colors, nsubdivs, zero_out_white=False, weights=None, normalize=False, squeeze=False):
    '''
    Computes 3D color histograms for N images and M patches in each image; returns a separate histogram for every patch.

    :param colors: has shape [nbatches, npatches, npatchcolors, 3] or [nbatches, ncolors, 3], must be normalized to 0..1.
    :param nsubdivs: # bins per each histogram dimension
    :param weights: has shape [nbatches, npatches, npatchcolors, 1] or [nbatches, ncolors, 1] is used to weigh the
        contribution of every color, if present
    :return: (hists, hists_flat)
    hists - has shape [nbatches, npatches, nsubdivs, nsubdivs, nsubdivs] with R,G,B bins respectively as last dimensions
    hists_flat - has shape [nbatches, npatches, nsubdivs ** 3]
    '''
    with tf.name_scope("histogram"):
        # Ensure input has the right shape
        tf_util.assert_shape(colors, shape_dict={-1: 3})
        if len(colors.shape) <= 3:
            input = tf.expand_dims(colors, axis=1)  # add patch dimension
        else:
            input = colors

        indices, hists_shape, debug_out = hist_quantize(input, nsubdivs)
        hists, hists_flat = hist_count(indices, hists_shape,
                                       normalize=normalize, weights=weights, zero_out_white= zero_out_white,
                                       squeeze=squeeze)

    debug_out["count_info"] = (indices, hists_shape)
    return hists, hists_flat, debug_out


def hist_quantize(input, nsubdivs):
    with tf.name_scope("hist_quantize"):
        # Quantize colors and clip to range
        quant = tf.clip_by_value(tf.cast(input * nsubdivs, dtype=tf.int32), 0, nsubdivs - 1)
        # Get batch and patch index
        bp = tf.meshgrid(tf.range(tf.shape(quant)[0]), tf.range(tf.shape(quant)[1]), tf.range(tf.shape(quant)[2]))

        # For indices we want:
        # [ batch#, patch#, Rbin, Gbin, Bbin] for nbatches * npatches * npatchcolors rows
        quant_resh = tf.reshape(quant, [-1, 3])
        # 2,1,0; 2,0,1; 0,2,1; 0,1,2;
        b_resh = tf.reshape(tf.transpose(bp[0], perm=[1,0,2]), [-1, 1])
        p_resh = tf.reshape(tf.transpose(bp[1], perm=[1,0,2]), [-1, 1])
        indices = tf.concat([b_resh, p_resh, quant_resh], axis=1)

        # Hist has shape [nbatches, npatches, nsubdivs, nsubdivs, nsubdivs]
        hists_shape = [tf.shape(input)[0], tf.shape(input)[1], nsubdivs, nsubdivs, nsubdivs]

    debug_out = { "quant" : quant,
                  "bp" : bp,
                  "quant_resh" : quant_resh,
                  "b_resh" : b_resh,
                  "p_resh" : p_resh }
    return indices, hists_shape, debug_out


def hist_count(indices, hists_shape, normalize, weights, zero_out_white=False, squeeze=True, use_max_normalization=True):
    with tf.name_scope("hist_count"):
        input_weights = None
        if weights is not None:
            if len(weights.shape) <= 3:
                input_weights = tf.expand_dims(weights, axis=1)
            else:
                input_weights = weights

        if input_weights is not None:
            update = tf.reshape(input_weights, [-1])
        else:
            update = tf.fill([tf.shape(indices)[0]], 1.0)

        hists = tf.scatter_nd(indices, update, shape=hists_shape)
        nsubdivs = hists_shape[-1]
        if zero_out_white:
            with tf.name_scope("zero_out_white"):
                multiplier = np.ones([1, 1, nsubdivs, nsubdivs, nsubdivs], dtype=np.float32)
                multiplier[0, 0, nsubdivs - 1, nsubdivs - 1, nsubdivs - 1] = 0.0
                hists = tf.multiply(hists, multiplier)
        hists_flat = tf.reshape(hists, [hists_shape[0], hists_shape[1], hists_shape[2] * hists_shape[3] * hists_shape[4]])

        if normalize:
            # Max normalization is different: if we used max_normalization on 32x32 patches to train the
            # histogram, we have to mimick the same behavior here now
            if not use_max_normalization:
                hists_norm_factor = tf.reduce_sum(hists_flat, axis=2)
                hists_flat = tf.divide(hists_flat, tf.expand_dims(hists_norm_factor, axis=2))
                hists = tf.divide(hists,
                                  tf.expand_dims(tf.expand_dims(tf.expand_dims(hists_norm_factor, axis=2), axis=3),
                                                 axis=4))
            elif weights is None:
                hists_norm_factor = tf.reduce_max(hists_flat, axis=2)
                hists_flat = tf.divide(hists_flat, tf.expand_dims(hists_norm_factor, axis=2))
                hists = tf.divide(hists,
                                  tf.expand_dims(tf.expand_dims(tf.expand_dims(hists_norm_factor, axis=2), axis=3), axis=4))
            else:
                # First we normalize by count
                hists_norm_factor = tf.reduce_sum(hists_flat, axis=2)
                hists_flat = tf.divide(hists_flat, tf.expand_dims(hists_norm_factor, axis=2))
                hists = tf.divide(hists,
                                  tf.expand_dims(tf.expand_dims(tf.expand_dims(hists_norm_factor, axis=2), axis=3),
                                                 axis=4))
                # Then multiply by 32*32 to get the same total count as an an unnormalized patch histogram
                LOG.warning('For weighted Max Histogram normalization assuming 32x32 scale')
                hists_flat = hists_flat * 32.0 * 32.0  # HACK: beware!
                hists = hists * 32.0 * 32.0

                # Then use max normalization
                hists_norm_factor = tf.reduce_max(hists_flat, axis=2)
                hists_flat = tf.divide(hists_flat, tf.expand_dims(hists_norm_factor, axis=2))
                hists = tf.divide(hists,
                                  tf.expand_dims(tf.expand_dims(tf.expand_dims(hists_norm_factor, axis=2), axis=3),
                                                 axis=4))
        if squeeze:
            hists = tf.squeeze(hists, axis=1)
            hists_flat = tf.squeeze(hists_flat, axis=1)

    return hists, hists_flat


# TODO: add an option to use LAB
def compute_patch_hist(images, hist_subdivs, patch_width, patch_stride=-1, zero_out_white=False, alphas=None):
    '''
    Composite function, which constructs an operation graph in tensorfow that:
    1. Extracts patches from a set of images (see get_patches)
    2. Computes a histogram for every patch (see compute_hist)
    3. Reduces all patch histograms for one image to one histogram with overall maximum in each bin
    4. Normalizes each reduced histogram by total count contained in it

    :param images: has shape [nbatches, height, width, 3]
    :param hist_subdivs: # bins per each histogram dimension
    :param patch_width: width of patch
    :param patch_stride: stride of patch
    :return: dictionary with keys
             'hist3d' : [nbatches, npatches, nsubdivs, nsubdivs, nsubdivs] with bins as last dimensions
             'max_hist3d' : [nbatches, nsubdivs, nsubdivs, nsubdivs] with maximal bin values across patches
             'hist' : [nbatches, npatches, nsubdivs x nsubdivs x nsubdivs] with histogram flattened
             'max_hist' : [nbatches, nsubdivs x nsubdivs x nsubdivs] with maximal bin values across patches
             'hist_sum' : [nbatches] with sum of all bins per image
             'norm_max_hist' : [nbatches, nsubdivs x nsubdivs x nsubdivs] with normalized maximal bin values across patches
             'patch_colors' : [ nbatches, npatches, pwidth x pwidth, 3 ] colors of image patches
             'patch_images' : [ nbatches, npatches, pwidth, pwidth, 3 ] patch images for debugging and visualization
    '''
    patches, patch_colors, patch_images = get_patches(images, pwidth=patch_width, stride=patch_stride)
    alpha_weights = None
    if alphas is not None:
        _, alpha_weights, _ = get_patches(alphas, pwidth=patch_width, stride=patch_stride)
    hist, hist_flat, deb = compute_hist(patch_colors, hist_subdivs, zero_out_white=zero_out_white, weights=alpha_weights)
    max_hist_flat = tf.reduce_max(hist_flat, axis=1)
    max_hist = tf.reduce_max(hist, axis=1)
    hist_sum = tf.reduce_sum(max_hist_flat, axis=1)
    norm_hist = tf.divide(max_hist_flat, tf.expand_dims(hist_sum, 1))

    return { 'hist3d' : hist,
             'max_hist3d' : max_hist,
             'hist' : hist_flat,
             'max_hist' : max_hist_flat,
             'hist_sum' : hist_sum,
             'norm_max_hist' : norm_hist,
             'patch_colors' : patch_colors,
             'patch_images' : patch_images,
             'count_info' : deb['count_info']
             }


def flat_hist_index(bin_i, bin_j, bin_k, nsubdivs):
    '''
    Returns index into flattened histogram for an i,j,k th bin in an nsubdiv 3D histogram
    '''
    return bin_i * nsubdivs * nsubdivs + bin_j * nsubdivs + bin_k

# RBF-BASED HISTOGRAM --------------------------------------------------------------------------------------------------


def create_hist_indexer(n_bins):
    '''
    Creates an "indexer" that helps bin colors for a histogram, by reshaping RGB
    value for the center of every nbins x nbins x nbins histogram cell.

    Result shape: 1, 1, 3, nbins^3
    '''
    res =  np.zeros([1, 1, 3, n_bins * n_bins * n_bins], dtype=np.float32)
    resR = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    resG = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    resB = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                resR[i][j][k] = (i + 0.5) / n_bins
                resG[i][j][k] = (j + 0.5) / n_bins
                resB[i][j][k] = (k + 0.5) / n_bins
    res[:, :, 0, :] = resR.reshape([1, n_bins * n_bins * n_bins])
    res[:, :, 1, :] = resG.reshape([1, n_bins * n_bins * n_bins])
    res[:, :, 2, :] = resB.reshape([1, n_bins * n_bins * n_bins])
    res_raw = np.zeros([n_bins, n_bins, n_bins, 3], dtype=np.float32)
    res_raw[:, :, :, 0] = resR
    res_raw[:, :, :, 1] = resG
    res_raw[:, :, :, 2] = resB
    return res_raw, res


# Computing a faster RBF histogram:
#
def create_hist_indexer_with_alpha(n_bins):
    '''
    '''
    res = np.zeros([1, 1, 4, n_bins * n_bins * n_bins], dtype=np.float32)
    resR = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    resG = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    resB = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    resA = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                resR[i][j][k] = (i + 0.5) / n_bins
                resG[i][j][k] = (j + 0.5) / n_bins
                resB[i][j][k] = (k + 0.5) / n_bins
                resA[i][j][k] = (n_bins - 0.5) / n_bins  # only last full alpha bin included
    res[:, :, 0, :] = resR.reshape([1, n_bins * n_bins * n_bins])
    res[:, :, 1, :] = resG.reshape([1, n_bins * n_bins * n_bins])
    res[:, :, 2, :] = resB.reshape([1, n_bins * n_bins * n_bins])
    res[:, :, 3, :] = resA.reshape([1, n_bins * n_bins * n_bins])
    res_raw = np.zeros([n_bins, n_bins, n_bins, 4], dtype=np.float32)
    res_raw[:, :, :, 0] = resR
    res_raw[:, :, :, 1] = resG
    res_raw[:, :, :, 2] = resB
    res_raw[:, :, :, 2] = resA
    return res_raw, res


def bin_colors(x, indexer, sigma_sq, weights=None):
    '''

    :param x:
    :param indexer:
    :param sigma_sq:
    :param weights:
    :return:
    '''

    # X has shape batchsize x w x h x RGB channels

    # have a pre-computed indexer 1 x 1 x 3 channels x xdims (RGB value of cell)
    # to compute contribution of colors to a given index, need:
    # distance of every color to every cell

    # indexer must have shape (1, 1, 3, xdims)
    #sh = [ d for d in x.shape.as_list()]
    #new_sh = [sh[0], sh[1] * sh[2], 3, 1]
    #LOG.debug('Binnable shape is %s' % str(sh))
    new_sh = [tf.shape(x)[0], -1, tf.shape(x)[-1], 1] #sh[1] * sh[2], 3, 1]
    LOG.debug('Desired shape is %s ' % str(new_sh))
    x_resh = tf.reshape(x, tf.stack(new_sh))
    LOG.debug('x_resh shape: %s ' % str(x_resh.shape))
    LOG.debug('indexer shape: %s' % str(indexer.shape))

    # diff will have size batchsize x (w * h) x 3 x (xdims)
    diff = tf.subtract(indexer, x_resh)
    LOG.debug('diff shape: %s' % str(diff.shape))

    # norm will have size batchsize x (w * h) x xdims
    norm = tf.reduce_sum(tf.pow(diff, 2.0), axis=2)
    LOG.debug('norm shape: %s' % str(norm.shape))

    # rbf will have size batchsize x (w * h) x xdims
    rbf = tf.exp(tf.divide(norm, -2 * sigma_sq))
    LOG.debug('rbf shape: %s' % str(rbf.shape))

    # TODO: Now we optionally weigh the rbf by weights
    if weights is not None:
        new_w_sh = [tf.shape(weights)[0], tf.shape(weights)[1], 1]
        weights_resh = tf.reshape(weights, new_w_sh)
        LOG.debug('weights shape: %s' % str(weights.shape))
        LOG.debug('weights reshaped shape: %s' % str(weights_resh.shape))
        rbf = tf.multiply(rbf, weights_resh)

    # hist will have size batchsize x xdims
    res = tf.reduce_sum(rbf, axis=1)
    LOG.debug('res shape: %s' % str(res.shape))

    return { 'img_resh': x_resh,
             'cell_diff': diff,
             'cell_norm': norm,
             'cell_rbf' : rbf,
             'bins' : res }


def compute_rbf_hist(colors, nsubdivs, sigma_sq, weights=None):
    '''
    Computes a histogram by modeling contributions of each color as a decaying Radial-Basis Function
    with gaussian kernel of variance of sigma_sq. If no weights are provided, treats each color as having
    a count of 1. Otherwise, multiplied each color's contributions by its weight.

    :param colors:
    :param nsubdivs:
    :param sigma_sq:
    :param weights:
    :return:
    '''
    if colors.shape[-1] == 3:
        _, ind = create_hist_indexer(nsubdivs)
    else:
        _, ind = create_hist_indexer_with_alpha(nsubdivs)
    indexer = tf.constant(ind)
    res = bin_colors(colors, indexer, sigma_sq, weights=weights)
    res['indexer'] = indexer
    res['bins_sum'] = tf.reduce_sum(res['bins'], axis=1)
    hist_norm = tf.divide(res['bins'], tf.reshape(res['bins_sum'], [-1, 1]))

    return hist_norm, res


# APPROXIMATIONS -------------------------------------------------------------------------------------------------------
def slice_rgb(colors):
    if colors.shape[-1] == 3:
        return colors
    else:
        rgb_shape = [x for x in tf.shape(colors)]
        rgb_shape[-1] = 3
        return tf.slice(colors, [0 for x in range(len(colors.shape))], rgb_shape)


def get_alpha_mask_recon_error(target_colors, source_colors, target_alphas=None, squared=False, blend=True):
    '''

    :param target_colors: [Nbatches, Ncolors, 3]
    :param source_colors: [Nbatches, Npalettes, Npalettecolors, 3]
    :param target_alphas: [Nbatches, Ncolors, Npalettes]
    :param use_lab: bool (whether to compute distnaces in LAB or RGB)
    :param squared: bool (whether to compare/output squared distances or L1)
    :param blend: bool (whether to reconstruct/get error by blending alpha or picking one best)
    :return:
    '''

    # diff: [Nbatches, Ncolors, Npalettes, Npalettecolors, 3]
    print('Target colors %s' % str(target_colors.shape))
    print('Source colors %s' % str(source_colors.shape))
    diff = tf.subtract(tf.expand_dims(tf.expand_dims(target_colors, axis=2), axis=2), tf.expand_dims(source_colors, axis=1))

    # norm: [Nbatches, Ncolors, Npalettes, Npalettecolors] - best color in each palette
    norm = tf.reduce_sum(tf.square(diff), axis=4)
    if not squared:
        eps = 1.0e-5  # keep derivative of square root stable
        norm = tf.sqrt(norm + eps)

    # Now we get best color in every palette
    # [Nbatches, Ncolors, Npalettes]
    best_pnorm = tf.reduce_min(norm, axis=3)
    print('best_pnorm: %s' % str(best_pnorm.shape))
    best_pcolor_idx = tf.cast(tf.argmin(norm, axis=3), tf.int32)
    print('best_pcolor_idx(before): %s' % str(best_pcolor_idx.shape))

    # [Nbatches, Ncolors], where [b,i] = b for all i
    bnums = tf.tile(tf.expand_dims(tf.range(tf.shape(target_colors)[0]), axis=1), [1, tf.shape(target_colors)[1]])
    print('Bnums: %s' % str(bnums.shape))
    cnums = tf.tile(tf.expand_dims(tf.range(tf.shape(target_colors)[1]), axis=0), [tf.shape(target_colors)[0], 1])
    print('Cnums: %s' % str(cnums.shape))

    # Now we select the best palette, or blend
    if not blend or target_alphas is None:
        if target_alphas is not None:
            # [Nbatches, Ncolors]
            best_palette = tf.cast(tf.argmax(target_alphas, axis=2), tf.int32)
        else:
            # [Nbatches, Ncolors]
            best_palette = tf.cast(tf.argmin(best_pnorm, axis=2), tf.int32)

        print('best palette: %s' % str(best_palette.shape))

        # Now select best color from best palette for norm and reconstruction
        pnorm_idx = tf.concat([tf.expand_dims(bnums, axis=2),
                               tf.expand_dims(cnums, axis=2),
                               tf.expand_dims(best_palette, axis=2)], axis=2)
        print('pnorm_idx: %s' % str(pnorm_idx.shape))
        err = tf.gather_nd(best_pnorm, pnorm_idx)  # [ Nbatches, Ncolors ]
        print('norm: %s' % str(err.shape))
        best_pcolor_idx = tf.gather_nd(best_pcolor_idx, pnorm_idx)  # [ Nbatches, Ncolors]
        print('best_pcolor_idx(after): %s' % str(best_pcolor_idx.shape))

        # Indexing into [Nbatches, Npalettes, Npalettecolors, 3]
        # Want: idx[b,c] = [b, best_palette, best_color]
        color_idx = tf.concat([tf.expand_dims(bnums, axis=2),
                               tf.expand_dims(best_palette, axis=2),
                               tf.expand_dims(best_pcolor_idx, axis=2)], axis=2)
        print('color_idx: %s' % str(color_idx.shape))
        approx = tf.gather_nd(source_colors, color_idx)  # [ Nbatches, Ncolors, 3 ]
        mapping = color_idx
    else:
        # To blend colors, we first select 1 color from each palette for each target pixel

        # [Nbatches, Ncolors, Npalettes, 3]  (batch#, palette#, palette_color#)
        bnums = tf.tile(tf.expand_dims(bnums, axis=2), [1, 1, tf.shape(target_alphas)[2]])
        pnums = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(target_alphas)[2]), axis=0), [tf.shape(target_colors)[1], 1]),
                                       axis=0),
                        [tf.shape(target_colors)[0], 1, 1])
        print('Bnums after: %s' % str(bnums.shape))
        print('Pnums after: %s' % str(pnums.shape))

        bc_idx = tf.concat([tf.expand_dims(bnums, axis=3),
                            tf.expand_dims(pnums, axis=3),
                            tf.expand_dims(best_pcolor_idx, axis=3)], axis=3)

        # [Nbatches, Ncolors, Npalettes, 3]
        blended_colors = tf.gather_nd(source_colors, bc_idx)
        print('blended_colors: %s' % str(blended_colors.shape))

        approx = tf.reduce_sum(blended_colors * tf.expand_dims(target_alphas, axis=3), axis=2)  # reduce out palettes
        #blended_colors = blended_colors * tf.expand_dims(target_alphas, axis=3)
        #approx = tf.slice(blended_colors, [0, 0, 1, 0], [tf.shape(target_colors)[0], tf.shape(target_colors)[1], 1, 3])
        #approx = tf.slice(blended_colors * tf.expand_dims(target_alphas, axis=3), [0, 0, 1, 0], [tf.shape(target_colors)[0], tf.shape(target_colors)[1], 1, 3])
        err = tf.reduce_sum(tf.square(approx - target_colors), axis=2)
        if not squared:
            eps = 1.0e-5  # keep derivative of square root stable
            err = tf.sqrt(err + eps)
        mapping = bc_idx

    return approx, err, mapping


def compute_alpha_mask_reconstruction(mapping, source_colors, target_alphas=None, blend=True):
    if not blend or target_alphas is None:
        approx = tf.gather_nd(source_colors, mapping)  # [ Nbatches, Ncolors, 3 ]
    else:
        blended_colors = tf.gather_nd(source_colors, mapping)
        approx = tf.reduce_sum(blended_colors * tf.expand_dims(target_alphas, axis=3), axis=2)  # reduce out palettes
    return approx


def get_best_color_recon_error(in_target_colors, in_source_colors, use_lab=False, use_source_alpha=False, squared=False):
    '''
    Returns best match for target colors from source_colors.
    :param target_colors: [Nbatches x n_target_colors x 3] RGB colors
    :param source_colors: [Nbatches x n_source_colors x 3 or 4] RGB colors
    :return:
    '''

    if use_lab:
        target_colors = tf.reshape(rgb2lab(tf.reshape(in_target_colors, [-1, 3])),
                                   tf.stack([tf.shape(in_target_colors)[0], tf.shape(in_target_colors)[1], 3]))
        source_colors = tf.reshape(rgb2lab(tf.reshape(slice_rgb(in_source_colors), [-1, 3])),
                                   tf.stack([tf.shape(in_source_colors)[0], tf.shape(in_source_colors)[1], 3]))
        if in_source_colors.shape[-1] == 4:
            source_colors = tf.concat(
                source_colors, tf.slice(in_source_colors, [0,0,3], [tf.shape(in_source_colors)[0], tf.shape(in_source_colors)[1], 1]))
    else:
        target_colors = in_target_colors
        source_colors = in_source_colors

    if use_source_alpha:
        ones = tf.ones([tf.shape(target_colors)[0], tf.shape(target_colors)[1], 1], dtype=tf.float32)
        target_colors = tf.concat([target_colors, (100.0 if use_lab else 1.0) * ones], axis=2)

    # diff: [Nbatches, Ncolors, Npalettecolors, 3]
    diff = tf.subtract(tf.expand_dims(target_colors, axis=2), tf.expand_dims(source_colors, axis=1))
    # norm: [Nbatches, Ncolors, Npalettecolors]
    norm = tf.reduce_sum(tf.square(diff), axis=3)
    if not squared:
        eps=1.0e-5  # keep derivative of square root stable
        norm = tf.sqrt(norm + eps)

    # best_norm: [Nbatches, Ncolors]
    best_norm = tf.reduce_min(norm, axis=2)
    best_idx = tf.cast(tf.argmin(norm, axis=2), tf.int32)
    # batch numbers used for indexing
    bnums = tf.tile(tf.expand_dims(tf.range(tf.shape(target_colors)[0]), axis=1), [1, tf.shape(target_colors)[1]])

    idx = tf.concat([tf.expand_dims(bnums, axis=2), tf.expand_dims(best_idx, axis=2)], axis=2)

    approx = tf.gather_nd(in_source_colors, idx)
    if use_source_alpha:
        approx = tf.slice(approx, [0,0,0], [tf.shape(in_target_colors)[0], tf.shape(in_target_colors)[1], 3])
    return approx, best_norm


def get_numpy_hashed_approximation(in_target_colors, in_source_colors, use_lab=False, use_source_alpha=False):
    if use_lab:
        raise RuntimeError('Lab not implemented')

    # Step 1: finely hash all the input colors
    nbins = 70.0
    quant = np.minimum(np.floor(in_target_colors * nbins), nbins - 1).astype(np.int32)
    hashes = np.left_shift(quant[:, 0], 20) + np.left_shift(quant[:,1], 10) + quant[:,2]

    uhashes, idx, idx_rec = np.unique(hashes,return_index=True,return_inverse=True)
    #LOG.debug('Computing reconstruction for %s (%s unique) from %s' %
    #          (str(in_target_colors.shape), str(uhashes.shape), str(in_source_colors.shape)))
    #print('Hashes size: %s (%s unique)' % (str(hashes.shape), str(uhashes.shape)))

    ucolors = in_target_colors[idx, :]
    #print('Ucolors %s' % str(ucolors.shape))
    if use_source_alpha:
        ucolors = np.concatenate([ucolors, np.ones([ucolors.shape[0], 1])], axis=1)

    diff = np.expand_dims(ucolors, axis=1) - np.expand_dims(in_source_colors, axis=0)
    #print('Diff %s' % str(diff.shape))

    norm = np.sqrt(np.sum(np.power(diff, 2.0), axis=2))
    #print('Norm %s' % str(norm.shape))

    best = np.argmin(norm, axis=1)
    best_idx = best[idx_rec]
    #print('Best %s' % str(best.shape))

    approx = np.zeros(in_target_colors.shape, dtype=np.float32)
    approx[:, :] = in_source_colors[best_idx, 0:3]
    #print('Recon %s' % str(approx.shape))

    metric = np.min(norm, axis=1)[idx_rec]
    #print('Norm %s' % str(metric.shape))
    #LOG.debug('Computed reconstruction')

    return approx.astype(np.float32), metric.astype(np.float32), best_idx.astype(np.int64)

# MISC -----------------------------------------------------------------------------------------------------------------
def make_onehot(alphas):
    '''
    Only keeps the largest value in the 3rd dimension; sets the rest to zero.
    :param alphas: [Nbatchs, ncolors, nchannels]
    :return: [Nbatches, ncolors, channels]
    '''

    # index into the last dimension
    best_palette = tf.cast(tf.argmax(alphas, axis=2), tf.int32)
    # index into batches
    bnums = tf.tile(tf.expand_dims(tf.range(tf.shape(alphas)[0]), axis=1), [1, tf.shape(alphas)[1]])
    # index into colors
    cnums = tf.tile(tf.expand_dims(tf.range(tf.shape(alphas)[1]), axis=0), [tf.shape(alphas)[0], 1])

    # we get the values we are interested in out
    best_idx = tf.concat([tf.expand_dims(bnums, axis=2),
                          tf.expand_dims(cnums, axis=2),
                          tf.expand_dims(best_palette, axis=2)], axis=2)
    update = tf.gather_nd(alphas, best_idx)

    # now we set these values in the result
    result = tf.scatter_nd(best_idx, update, tf.shape(alphas))
    return result


HACKY_MEAN_COLORS = np.array([[0.3302, 0.3249, 0.3381],
                              [0.1057, 0.1003, 0.1033],
                              [0.7625, 0.2766, 0.1564],
                              [0.2660, 0.3675, 0.6394],
                              [0.9141, 0.7677, 0.3350],
                              [0.8174, 0.4739, 0.5425],
                              [0.9102, 0.9029, 0.8971],
                              [0.3376, 0.1752, 0.1565],
                              [0.5987, 0.6575, 0.6664],
                              [0.4212, 0.6110, 0.2949],
                              [0.8979, 0.7821, 0.6204],
                              [0.5996, 0.4602, 0.3265]], dtype=np.float32)
