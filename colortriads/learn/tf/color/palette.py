import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import util.img_util as img_util
from PIL import Image,ImageDraw

import util.plot_util as plot_util
from learn.tf.log import LOG
import learn.tf.color.color_ops as color_ops

global_triangles = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 1],
        [1, 8, 2], [2, 8, 9], [2, 9, 10], [2, 10, 3], [3, 10, 11], [3, 11, 12],
        [3, 12, 4], [12, 13, 4], [4, 13, 14], [4, 14, 5], [5, 14, 15], [5, 15, 16],
        [6, 5, 16], [6, 16, 17], [18, 6, 17], [18, 1, 6], [7, 1, 18], [1, 7, 8]]

def __init_vertices():
    vertices = np.zeros([19, 2], np.float32)
    sq3_quarter = math.sqrt(3.0) * 0.25
    sq3_half = sq3_quarter * 2

    # Handle Y
    for idx in [11, 10, 9]: vertices[idx, 1] = sq3_half
    for idx in [2, 3, 8, 12]: vertices[idx, 1] = sq3_quarter
    for idx in [0, 1, 4, 7, 13]: vertices[idx, 1] = 0
    for idx in [14, 5, 6, 18]: vertices[idx, 1] = -sq3_quarter
    for idx in [15, 16, 17]: vertices[idx, 1] = -sq3_half

    # Handle X
    for idx in [0, 10, 16]: vertices[idx, 0] = 0
    for idx in [3, 5]: vertices[idx, 0] = -0.25
    for idx in [11, 4, 15]: vertices[idx, 0] = -0.5
    for idx in [12, 14]: vertices[idx, 0] = -0.75
    for idx in [13]: vertices[idx, 0] = -1.0
    for idx in [2, 6]: vertices[idx, 0] = 0.25
    for idx in [9, 1, 17]: vertices[idx, 0] = 0.5
    for idx in [8, 18]: vertices[idx, 0] = 0.75
    for idx in [7]: vertices[idx, 0] = 1.0
    return vertices

global_vertices = __init_vertices()


def get_activated_triangles(n_vertices):
    '''
    Returns triangles activated by this many vertices
    :param n_vertices:
    :param n_tri_subdivs:
    :return:
    '''
    triangles = []
    for i in range(len(global_triangles)):
        tri = global_triangles[i]
        if tri[0] < n_vertices and tri[1] < n_vertices and tri[2] < n_vertices:
            triangles.append(tri)
    return triangles


def create_rgb_tri_interpolator_linear(n_vertices, n_tri_subdivs, return_right_side_up_triangles=False):
    # Which triangles to interpolate?
    triangles = []
    for i in range(len(global_triangles)):
        tri = global_triangles[i]
        if tri[0] < n_vertices and tri[1] < n_vertices and tri[2] < n_vertices:
            triangles.append(tri)

    # Compute various interpolation properties
    n_tri = len(triangles)
    n_tri_colors = n_tri_subdivs ** 2
    n_tri_rows = 2 * n_tri_subdivs - 1

    # Compute all interpolations for a generic triangle
    right_side_up_indicator = np.zeros([n_tri_colors], dtype=np.bool)
    weights = np.zeros([3, n_tri_colors], dtype=np.float32)
    cweights = np.zeros([3, n_tri_colors * 3], dtype=np.float32)  # coordinates, for viz
    color_count = 0

    tri_centroids = np.zeros([n_tri_colors, 2], dtype=np.float32)
    tri_interps = np.zeros([n_tri_colors, 2], dtype=np.float32)

    def compute_weights(r, c):
        alpha = r / (n_tri_rows - 1.0)
        n_cols = n_tri_rows - (r / 2) * 2
        if n_cols == 1:
            beta = 0.5
        else:
            beta = c / (n_cols - 1.0)
        return alpha, ((1 - alpha) * beta), ((1 - alpha) * (1 - beta))

    for r in range(0, n_tri_rows):
        alpha = r / (n_tri_rows - 1.0)
        n_cols = n_tri_rows - (r / 2) * 2 #(n_tri_rows - r + 1) / 2
        start_col = r % 2 #(r + 1) / 2
        for c in range(int(start_col), int(n_cols - start_col), 2):
            if (r % 2) == 0:
                if n_cols == 1:
                    beta = 0.5
                else:
                    beta = c / (n_cols - 1.0)
                weights[2, color_count] = alpha
                weights[1, color_count] = (1 - alpha) * beta
                weights[0, color_count] = (1 - alpha) * (1 - beta)
                right_side_up_indicator[color_count] = True
            else:
                # Just average neighboring weights
                # bottom left neighbor
                w0 = compute_weights(r - 1, c - 1)
                w1 = compute_weights(r - 1, c + 1)
                w2 = compute_weights(r + 1, c - 1)
                weights[2, color_count] = (w0[0] + w1[0] + w2[0]) / 3.0
                weights[1, color_count] = (w0[1] + w1[1] + w2[1]) / 3.0
                weights[0, color_count] = (w0[2] + w1[2] + w2[2]) / 3.0
            tri_interps[color_count, :] = [weights[0, color_count], weights[1, color_count]]

            # coordinates for vizualization------------
            viz_row = (r / 2) + 0.0
            if (r % 2) == 0:  # right side up
                # Lower left
                calpha = (viz_row + 0.0) / n_tri_subdivs
                cbeta = (c/2) / max(1.0, n_tri_subdivs - viz_row) #(abs_col + 0.0) / n_tri_cols
                cweights[2, color_count * 3] = calpha
                cweights[1, color_count * 3] = (1 - calpha) * cbeta
                cweights[0, color_count * 3] = (1 - calpha) * (1 - cbeta)
                # Lower right
                calpha = (viz_row + 0.0) / n_tri_subdivs
                cbeta = (c/2 + 1.0) / max(1.0, n_tri_subdivs - viz_row) #(abs_col + 2.0) / n_tri_cols
                cweights[2, color_count * 3 + 1] = calpha
                cweights[1, color_count * 3 + 1] = (1 - calpha) * cbeta
                cweights[0, color_count * 3 + 1] = (1 - calpha) * (1 - cbeta)
                # Top
                calpha = (viz_row + 1.0) / n_tri_subdivs
                cbeta = (c/2 + 0.0) / max(1.0, n_tri_subdivs - viz_row - 1) #(abs_col + 1.0) / n_tri_cols
                cweights[2, color_count * 3 + 2] = calpha
                cweights[1, color_count * 3 + 2] = (1 - calpha) * cbeta
                cweights[0, color_count * 3 + 2] = (1 - calpha) * (1 - cbeta)
            else:
                # Upper left
                calpha = (viz_row + 1.0) / n_tri_subdivs
                cbeta = (c/2) / max(1.0, n_tri_subdivs - viz_row - 1)
                cweights[2, color_count * 3] = calpha
                cweights[1, color_count * 3] = (1 - calpha) * cbeta
                cweights[0, color_count * 3] = (1 - calpha) * (1 - cbeta)
                # Upper right
                calpha = (viz_row + 1.0) / n_tri_subdivs
                cbeta = (c/2 + 1.0) / max(1.0, n_tri_subdivs - viz_row - 1)
                cweights[2, color_count * 3 + 1] = calpha
                cweights[1, color_count * 3 + 1] = (1 - calpha) * cbeta
                cweights[0, color_count * 3 + 1] = (1 - calpha) * (1 - cbeta)
                # Bottom
                calpha = (viz_row + 0.0) / n_tri_subdivs
                cbeta = (c/2 + 1.0) / max(1.0, n_tri_subdivs - viz_row)
                cweights[2, color_count * 3 + 2] = calpha
                cweights[1, color_count * 3 + 2] = (1 - calpha) * cbeta
                cweights[0, color_count * 3 + 2] = (1 - calpha) * (1 - cbeta)

            # Centroid is the average of vertex positions
            center_alpha = np.sum(cweights[0, (color_count * 3):(color_count * 3 + 3)]) / 3.0
            center_beta = np.sum(cweights[1, (color_count * 3):(color_count * 3 + 3)]) / 3.0
            tri_centroids[color_count, :] = [center_alpha, center_beta]

            color_count += 1

    # Fill in all interpolated colors for each active triangle
    res = np.zeros([n_tri * n_tri_colors, n_vertices], dtype=np.float32)
    verts = np.zeros([n_tri * n_tri_colors, 3, 2], dtype=np.float32)
    for t in range(0, n_tri):
        tri = triangles[t]
        for c in range(n_tri_colors):
            for i in range(0, 3):
                # for every vertex of triangle, update its weight for this tri
                res[t * n_tri_colors + c, tri[i]] = weights[i, c]
                # for every vertex of a color's triangle, set its value in the verts matrix
                verts[t * n_tri_colors + c, i, :] = cweights[:, c * 3 + i].dot(global_vertices[tri])

    if return_right_side_up_triangles:
        return res, verts, n_tri, cweights, tri_centroids, tri_interps, right_side_up_indicator
    else:
        return res, verts, n_tri, cweights, tri_centroids, tri_interps


def compute_bernstein_polynomials(interpolator):
    '''

    :param interpolator: [P x 3]
    :return: [P x 10], where 8 bernstein polynomial weights are for a cubic bezier triangle and correspond to points
    p300, p030, p003, p012, p021, p102, p201, p120, p210, p111 in that order
    '''
    def compute_bernstein(i, j, k, interp):
        n = 3
        B = (math.factorial(n) + 0.0) / (math.factorial(i) * math.factorial(j) * math.factorial(k))
        return B * np.power(interp[:, 0], i) * np.power(interp[:, 1], j) * np.power(interp[:, 2], k)

    res = np.concatenate([np.expand_dims(compute_bernstein(3, 0, 0, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(0, 3, 0, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(0, 0, 3, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(0, 1, 2, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(0, 2, 1, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(1, 0, 2, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(2, 0, 1, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(1, 2, 0, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(2, 1, 0, interpolator), axis=1),
                          np.expand_dims(compute_bernstein(1, 1, 1, interpolator), axis=1)], axis=1)
    return res


def create_rgb_tri_interpolator(n_vertices, n_tri_subdivs, use_barycentric_color_interp=False):
    '''
    Create a deterministic color interpolator matrix for a fixed palette layout.

    :param n_vertices: number of vertices in the hex palette template to activate
    :param n_tri_subdivs: maximum number of subdivisions per triangle,
        with 1 subdivision corresponding to 4 subtriangles in each palette triangle, 2 subdivisions - to 16 subtriangels
    :return:
    res - [ N_triangles * N_triangle_colors, N_palette_colors] interpolation weight matrix
    verts - [ N_triangles * N_triangle_colors, 3, 2] positions of vertices of each subtriangle (TODO: change for levels)
    n_tri - number of triangles activated due to n_vertices
    tiers_mat - [n_tri * n_tri_colors, n_tri_subdivs + 1, n_tri] - indicator matrix for each triangle
    tiers_idx - [n_tri * n_tri_colors] - integer indicating the patch's subdivision index
    tri_idx - [n_tri * n_tri_colors] - integer indicating patch's triangle's index in the palette
    '''
    # Which triangles to interpolate?
    triangles = get_activated_triangles(n_vertices)

    # Compute various interpolation properties
    n_tri = len(triangles)
    n_tri_colors = 4 ** n_tri_subdivs
    n_tri_rows = 2 ** n_tri_subdivs

    n_tri_tri = int(4 * (4.0 ** n_tri_subdivs - 1.0) / (4.0 - 1))  # sum of geometric series 4 + 16 + 64 + ...

    # Compute all interpolations for a generic triangle
    # weights[i, c] = weight of ith palette color vertex for cth color
    weights = np.zeros([3, n_tri_tri], dtype=np.float32)
    # cweights[i, c*3 + v] = weight of ith palette vertex position for position of vth vertex of cth color patch
    cweights = np.zeros([3, n_tri_tri * 3], dtype=np.float32)  # coordinates, for viz
    # ctiers[c] = subdivision level of the ith patch
    ctiers = np.zeros([n_tri_tri], dtype=np.uint32)
    # Ids of triangle colors in the subdivied triangles
    tri_color_ids = np.zeros([n_tri_colors], dtype=np.uint32)

    # Level 0: we add colors of the triangle vertices themselves
    weights[0:3, 0] = [1, 0, 0] #[2.0/3.0, 1/6.0, 1/6.0]#
    weights[0:3, 1] = [0, 1, 0] #[1/6.0, 2.0/3.0, 1/6.0]#
    weights[0:3, 2] = [0, 0, 1] #[1/6.0, 1/6.0, 2/3.0]#
    ctiers[0:3] = 0  # subdivision tier 0
    # triangle vertices
    cweights[0:3, 0] = [1, 0, 0]
    cweights[0:3, 1] = [0.5, 0.5, 0]
    cweights[0:3, 2] = [0.5, 0, 0.5]
    cweights[0:3, 3 + 1] = [0, 1, 0]
    cweights[0:3, 3 + 2] = [0, 0.5, 0.5]
    cweights[0:3, 3 + 0] = [0.5, 0.5, 0]
    cweights[0:3, 3 * 2 + 2] = [0, 0, 1]
    cweights[0:3, 3 * 2 + 0] = [0.5, 0, 0.5]
    cweights[0:3, 3 * 2 + 1] = [0, 0.5, 0.5]

    # Level 1: we add the centroid
    weights[0:3, 3] = [1.0/3, 1.0/3, 1.0/3]  # even mix of vertex colors
    ctiers[3] = 1
    cweights[0:3, 3 * 3 + 0] = [0.5, 0.5, 0]
    cweights[0:3, 3 * 3 + 1] = [0, 0.5, 0.5]
    cweights[0:3, 3 * 3 + 2] = [0.5, 0, 0.5]

    tri_color_ids[0:4] = [0, 1, 2, 3]

    # We push these triangles onto the triangle stack and subdivide them
    triangle_stack = [ 0, 1, 2, 3 ]
    color_count = 4
    tri_tri_count = 4
    ALMOST_ONE = 0.99999
    for s in range(2, n_tri_subdivs + 1):
        new_triangle_stack = []
        for t in triangle_stack:
            # Barycentric coordinates of the new triangle centers (for color interpolation)
            color_bary = [[2 / 3.0, 1 / 6.0, 1 / 6.0], [1 / 6.0, 2 / 3.0, 1 / 6.0],
                          [1 / 6.0, 1 / 6.0, 2 / 3.0], [1 / 3.0, 1 / 3.0, 1 / 3.0]]
            add_ids = [0, 1, 2, 3]
            # Position weights of new triangle vertices
            new_cweights_idx = [ [0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5] ]
            new_cweights = np.zeros([3, 6], dtype=np.float32)
            new_cweights[:, 0] = cweights[:, t * 3]
            new_cweights[:, 1] = cweights[:, t * 3 + 1]
            new_cweights[:, 2] = cweights[:, t * 3 + 2]
            new_cweights[:, 3] = (cweights[:, t * 3] + cweights[:, t * 3 + 1]) / 2.0
            new_cweights[:, 4] = (cweights[:, t * 3 + 1] + cweights[:, t * 3 + 2]) / 2.0
            new_cweights[:, 5] = (cweights[:, t * 3] + cweights[:, t * 3 + 2]) / 2.0

            # If this is a corner triangle, we subdivide it differently
            # (to ensure that the color of the vertex color itself is always present in the palette
            max_idx = (weights[0:3, t]).argmax()
            if weights[max_idx, t] >= ALMOST_ONE:
                #max_idx = 0
                color_bary[max_idx] = [0, 0, 0]
                color_bary[max_idx][max_idx] = 1.0
                #new_cweights_idx[max_idx] = [3, 4, 5]
                add_ids.remove(max_idx) # = [1, 2, 3]
            else:
                add_ids.remove(3)

            # Add the 3 new triangles
            for nt in range(0, len(color_bary)):  # new triangle
                new_triangle_stack.append(tri_tri_count)
                bary = color_bary[nt]
                cw_idx = new_cweights_idx[nt]
                ctiers[tri_tri_count] = s
                for i in range(0, 3):  # update the weight of ith vertex's color
                    for j in range(0, 3):  # by summing up ith vertex's color contributions
                        weights[i, tri_tri_count] += bary[j] * cweights[i, t * 3 + j]
                for i in range(0, 3):  # update vth vertex weights by selecting from a list
                    cweights[:, tri_tri_count * 3 + i] = new_cweights[:, cw_idx[i]]
                if nt in add_ids:
                    tri_color_ids[color_count] = tri_tri_count
                    color_count += 1
                tri_tri_count += 1
        triangle_stack = new_triangle_stack

    # Fill in all interpolated colors for each active triangle
    res = np.zeros([n_tri * n_tri_colors, n_vertices], dtype=np.float32)
    # Fill in 3 vertices for each triangle in order to visualize
    verts = np.zeros([n_tri * n_tri_colors, 3, 2], dtype=np.float32)
    # Fill in tiers for each triangle as a matrix
    tiers_mat = np.zeros([n_tri * n_tri_colors, n_tri_subdivs + 1, n_tri], dtype=np.float32)
    tiers_idx = np.zeros([n_tri * n_tri_colors], dtype=np.int32)
    tri_idx = np.zeros([n_tri * n_tri_colors], dtype=np.int32)

    for t in range(0, n_tri):
        tri = triangles[t]
        for c in range(n_tri_colors):
            w_idx = tri_color_ids[c]
            color_idx = t * n_tri_colors + c
            tiers_mat[color_idx, ctiers[w_idx], t] = 1.0  # Only one element is activated per color
            tiers_idx[color_idx] = ctiers[w_idx]
            tri_idx[color_idx] = t
            for i in range(0, 3):
                # for every vertex of triangle, update its weight for this tri
                res[color_idx, tri[i]] = weights[i, w_idx]
                # for every vertex of a color's triangle, set its value in the verts matrix
                verts[color_idx, i, :] = cweights[:, w_idx * 3 + i].dot(global_vertices[tri])

    return res, verts, n_tri, tiers_mat, tiers_idx, tri_idx


def compute_subdivision_level_areas(level_values, n_tri, max_tri_subdivs):
    '''

    :param level_values:
    :param n_tri:
    :param max_tri_subdivs:
    :return:
    '''
    # map to 0...max_subdivs range
    levels = tf.multiply(level_values, max_tri_subdivs)

    # shape functions are of the form 4^-|x - tri_level| - tri_level
    # with 0th level shape function differing by having |x-f(x)|, where
    # f(x) = -e^-(x^2/(2*(0.4)^2)) in order to model 3-way splitting for level 0 subdivision and
    # 4-way splitting for level 1 subdivision
    ones = tf.ones([1, max_tri_subdivs + 1, 1])
    ones_1 = tf.ones([1, max_tri_subdivs, 1]);
    range0_n = np.array([range(0, int(max_tri_subdivs) + 1)], np.float32).reshape([1, -1, 1])
    range1_n = np.array([range(1, int(max_tri_subdivs) + 1)], np.float32).reshape([1, -1, 1])
    lev_exp = tf.expand_dims(levels, axis=1)
    exponent = ones * lev_exp
    exponent = exponent - range0_n
    #exponent2 = tf.abs(exponent1) * -1.0 - range0_n

    #exponent1_2 = ones_1 * lev_exp - range1_n
    #fx = levels + tf.exp(tf.divide(tf.pow(levels, 2.0), -2 * 0.4 * 0.4));
    #fx_exp = tf.expand_dims(fx, axis=1);
    #exponent_2 = tf.concat([fx_exp, exponent1_2], axis=1)

    final_exponent = tf.abs(exponent) * -1.0 - range0_n
    areas = tf.minimum(1.0/3, tf.pow(4.0, final_exponent))

    return areas

def visualize_palette_matplotlib(frag_colors, frag_verts, img_width):
    fig = plt.figure()
    for i in range(0, frag_colors.shape[0]) :
        color = frag_colors[i]
        plot_util.matplot_draw_triangle(fig, frag_verts[i, :, :], color)
    return plot_util.matplot_fig2img(fig, unit_cube=True, no_axes=True, width=img_width, height=img_width)


def equi_tri_area(v0, v1):
    '''
    Computes area of an equilateral triangle on a plane, or a set of N triangles.

    :param v0: 2-elem array for one vertex, or [Nx2] array for N triangles
    :param v1: 2-elem array for another vertex, or [Nx2] array for N triangles

    :return: area or [N] areas for a set of triangles
    '''
    d_sq = np.sum((v0 - v1) ** 2, axis=(1 if len(v0.shape) == 2 else 0))
    # Area = 0.5 a * 0.5 a * sqrt(3) = 0.25 * sqrt(3) * d_sq
    return 0.25 * math.sqrt(3.0) * d_sq


def transform_equi_tri_to_area(verts, area):
    '''
    Transforms a triangle (or a set of triangles) to have the same centroid(s) the given area(s).

    :param verts: [3x2] array of 3 vertices of an equilateral triangle, or [Nx3x2] for N triangles
    :param area: desired area of this triangle, or [N] areas for N triangles
    :return: [3x2] vertices of a triangle with same centroid and desired area, or [Nx3x2] for N triangles
    '''
    verts_shape = verts.shape
    if len(verts_shape) == 2:
        verts = np.expand_dims(verts, axis=0)

    # Center of the triangle
    c = np.sum(verts, axis=1) / 3.0
    c = np.expand_dims(c, axis=1)

    # Normal vectors from centroid to vertices
    diff = verts - c
    n = diff / np.expand_dims(np.sqrt(np.sum(diff ** 2.0, axis=2)), axis=2)

    # Half-side of the equilateral triangle with the desired area
    a = np.sqrt(area * 4 / math.sqrt(3.0)) * 0.5

    # Displacement from centroid along normal vectors to vertices to achieve this half-side
    d = 2.0 * a / math.sqrt(3.0)
    d = np.expand_dims(np.expand_dims(d, axis=1), axis=1)

    # Compute new verts
    new_verts = c + n * d
    return new_verts.reshape(verts_shape)


def visualize_palette_cv(img_width, patch_colors, patch_verts, tier_idx=None, tiers_to_render=None,
                         patch_areas=None, patch_levels=None, skip_rendering_higher_levels=False, zoom_in=True,
                         wind_value=None, wind_location=None):
    '''
    Visualizes a set of palette fragments (patches) as an image.

    :param patch_colors: [Nx3] array with RGB colors per each triangular palette fragment (or [Nx4] if alpha is there)
    :param patch_verts: [Nx3x2] array with x,y position of each triangular fragment
    :param img_width: int, pixel width to render at
    :param tier_idx: [N] uint32 array with the subdivision tier of each fragment; if present, renders lower tiers first
    :param tiers_to_render: uint32 list []; if present, only renders these tiers
    :param patch_areas: [N] float; if present, adjusts each fragment's size to equal area
    :param skip_rendering_higher_levels: bool, if true does not render levels lower than triangle's level

    :return: img_width x img_width x 3 image array
    '''
    nchannels = patch_colors.shape[1]
    if nchannels == 4:
        bgdata = img_util.create_checkered_image(img_width, img_width/30, color1=[0.9, 0.9, 0.9, 1.0], color2=[1.0, 1.0, 1.0, 1.0])
    else:
        bgdata = np.zeros((img_width, img_width, 4), np.float32) + 1.0
    bg_img = Image.fromarray((bgdata * 255).astype(np.uint8), 'RGBA')
    img = Image.new('RGBA', (img_width, img_width))
    drawer = ImageDraw.Draw(img)

    def to_pil_color(color_arr):
        return (int(color_arr[0] * 255),
                int(color_arr[1] * 255),
                int(color_arr[2] * 255),
                int(color_arr[3] * 255) if nchannels > 3 else 255)

    def to_pil_vertics(vert_array):
        return [(vert_array[i, 0], vert_array[i, 1]) for i in range(3)]

    # if areas are set, we transform triangles to respect areas
    if patch_areas is not None and tier_idx is not None:
        new_frag_verts = transform_equi_tri_to_area(patch_verts, patch_areas)
        # Keep level 0 vertices
        new_frag_verts[tier_idx == 0, :, :] = patch_verts[tier_idx == 0, :, :]
        # Keep size of patches for lower levels (scaling down does not work)
        if patch_levels is not None:
            new_frag_verts[tier_idx < patch_levels, :, :] = patch_verts[tier_idx < patch_levels, :, :]
        patch_verts = new_frag_verts

    if skip_rendering_higher_levels and patch_levels is not None and tier_idx is not None:
        patch_verts[tier_idx > patch_levels, :, :] = 0.0

    # Used for wind location
    first_tri_verts = np.expand_dims(np.array([[0, 0], [0.5, 0], [0.25, math.sqrt(3) * 0.25]], np.float32), axis=0)
    if zoom_in:
        max_coord = np.max(np.max(patch_verts, axis=1), axis=0)
        min_coord = np.min(np.min(patch_verts, axis=1), axis=0)
        scale = np.max(np.abs(max_coord - min_coord))
        desired_scale = 2.0
        scale_factor = desired_scale / scale
        patch_verts = (patch_verts - np.expand_dims(np.expand_dims(min_coord, axis=0), axis=0)) * scale_factor - 1.0
        first_tri_verts = (first_tri_verts - np.expand_dims(np.expand_dims(min_coord, axis=0), axis=0)) * scale_factor - 1.0

    # convert [-1,1] to [0, img_width-1]
    img_verts = np.maximum(0, np.minimum(img_width - 1, np.floor((patch_verts + 1.0) / 2.0 * img_width))).astype(np.int32)
    img_verts[:,:,1] = img_width - 1.0 - img_verts[:, :, 1]

    if tier_idx is not None:
        # Render from back to front
        _tiers_to_render = tiers_to_render
        if tiers_to_render is None:
            _tiers_to_render = range(0, max(tier_idx) + 1)
        #print('Patch colors shape %s' % str(patch_colors.shape))
        color_idx = np.array(range(0, patch_colors.shape[0]), dtype=np.uint32)
        #print('Tier idx shape %s' % str(tier_idx.shape))
        #print('Color idx shape %s' % str(color_idx.shape) )
        for s in _tiers_to_render:
            colors_in_tier = color_idx[tier_idx==s]
            #print('%d colors in tier %d' % (colors_in_tier.shape[0], s))
            #print('Colors in tier: %s' % str(colors_in_tier))
            for i in colors_in_tier:
                color = patch_colors[i]
                #print('Color poly %s %s' % (str(color.shape), str(color.tolist())))
                #print('Verts %s' % str(img_verts[i, :, :]))
                # TODO: there is a bug when only level 1 is shown, colors are wrong -- fix.
                #cv2.fillPoly(img, [img_verts[i, :, :]], color.tolist())
                drawer.polygon(to_pil_vertics(img_verts[i, :, :]), fill=to_pil_color(color.tolist()))
    else:
        for i in range(0, patch_colors.shape[0]) :
            color = patch_colors[i]
            drawer.polygon(to_pil_vertics(img_verts[i, :, :]), fill=to_pil_color(color.tolist()))

    if wind_location is not None:
        wloc = wind_location[0] * first_tri_verts[0,0,:] + wind_location[1] * first_tri_verts[0,1,:] + \
               (1.0 - wind_location[0] - wind_location[1]) * first_tri_verts[0,2,:]
        img_wloc = np.maximum(0, np.minimum(img_width - 1, np.floor((wloc + 1.0) / 2.0 * img_width))).astype(np.int32)
        img_wloc[1] = img_width - 1.0 - img_wloc[1]
        def _clipi(x):
            return int(min(img_width-1, max(0, x)))
        radius = img_width / 25
        bbox = (_clipi(img_wloc[0] - radius), _clipi(img_wloc[1] - radius),
                _clipi(img_wloc[0] + radius), _clipi(img_wloc[1] + radius))
        drawer.ellipse(bbox, fill = 'black', outline ='white')

    if wind_value is not None:
        wwidth = int(abs(wind_value) * img_width / 2)
        wbasecolor = np.array([1.0, 0, 0] if wind_value > 0 else [0, 0, 1.0])
        wcolor = abs(wind_value) * wbasecolor + (1.0 - abs(wind_value)) * np.array([1.0, 1.0, 1.0])
        wheight = int(img_width * (1 - 0.5 * math.sqrt(3)) / 2)
        if wwidth > 0:
            if wind_value > 0:
                bbox = (img_width/2, 0, img_width/2 + wwidth, wheight)
            else:
                bbox = (img_width / 2 - wwidth, 0, img_width / 2, wheight)
            drawer.rectangle(bbox, fill = to_pil_color(wcolor.tolist()))

    composite = Image.alpha_composite(bg_img, img)
    imdata = np.asarray(composite, dtype=np.uint8)

    return imdata[:,:,0:3].astype(np.float32) / 255.0

def visualize_palette_and_image(img, *args, **nargs):
    if len(img.shape) == 3 and img.shape[2] == 3:  # 3 channel image
        img_height = img.shape[0]
        result = np.zeros((img_height, img.shape[1] + img_height, 3), np.float32) + 1.0
        result[:,0:img.shape[1],:] = img
        result[:,img.shape[1]:,:] = visualize_palette_cv(img_height, *args, **nargs)
        return result
    else:
        # Image is not an image, just visualize palette
        return visualize_palette_cv(300, *args, **nargs)


def color_sail_to_string(colors, patchwork, wind=None):
    '''
    colors:
    :param colors: [nvertices x nchannels]
    :param wind: [3] [w pu pv]
    :param patchwork: int
    :return:
    '''
    def _colorstr(c):
        return '%0.4f %0.4f %0.4f' % (c[0], c[1], c[2])

    in_wind = wind
    if wind is None:
        in_wind = [0.0, 1.0/3.0, 1.0/3.0]

    return '%s %s %s %0.4f %0.4f %0.4f %d' % (_colorstr(colors[0,:]), _colorstr(colors[1,:]), _colorstr(colors[2,:]),
                                              in_wind[0], in_wind[1], in_wind[2], patchwork)


def color_sail_from_string(instr):
    parts = instr.split()
    colors = [ float(x) for x in parts[:9] ]
    wind = [ float(x) for x in parts[9:-1] ]
    wind.append(0)
    nsubdivs = int(parts[-1])

    colors = np.array(colors, dtype=np.float32).reshape(-1, 3)
    wind = np.array(wind, dtype=np.float32)
    return colors, wind, nsubdivs


def color_sail_to_float32_arr(colors, patchwork, wind=None):
    in_wind = wind
    if wind is None:
        in_wind = [0.0, 1.0 / 3.0, 1.0 / 3.0]

    return np.concatenate(
        [colors[0,:], colors[1,:], colors[2,:], in_wind, np.array([patchwork], np.float32)])



class PaletteOptions(object):
    def __init__(self, max_colors, max_tri_subdivs, wind_nchannels=0, discrete_continuous=False, use_alpha=False):
        self.n_colors = max_colors
        self.max_tri_subdivs = max_tri_subdivs
        self.discrete_continuous = discrete_continuous
        self.use_alpha = use_alpha
        self.n_channels = 4 if use_alpha else 3
        self.wind_nchannels = wind_nchannels
        self.wind_alpha = 0.8
        self.max_wind = 0.5

        if wind_nchannels > 0 and max_colors != 3:
            raise RuntimeError("Wind is only supported for 3-color sail palettes")


class PaletteHelper(object):
    def __init__(self, opts):
        self.opts = opts
        if self.opts.discrete_continuous:
            print('Discrete continuous---------------------------')
            self.interpolator, self.verts, self.n_tri, self.tiers_mat, self.tiers_idx, self.tri_idx = \
                create_rgb_tri_interpolator(self.opts.n_colors, self.opts.max_tri_subdivs)
            self.patch_center_uv = None
            self.patch_uv = None
            self.tri_right_side_up_indicator = None
        else:
            self.tiers_mat = None
            self.tiers_idx = None
            self.tri_idx = None
            self.interpolator, self.verts, self.n_tri, _, self.patch_center_uv, self.patch_uv, self.tri_right_side_up_indicator = \
                create_rgb_tri_interpolator_linear(self.opts.n_colors, self.opts.max_tri_subdivs, return_right_side_up_triangles=True)

        self.bernstein = compute_bernstein_polynomials(self.interpolator)

        # [ Nbatches x Nvertices x Nchannels ]
        self.colors = None

        # [ Nbatches x ?? ]
        self.areas = None
        self.levels = None

        # [ Nbatches x Ninterpcolors x Nchannels ]
        self.patch_colors = None

        # [ Nbatches x Ninterpcolors x 3 ]
        self.lab_patch_colors = None

        self.patch_areas = None

        # Wind stuff
        self.wind_val = None  # wind  [Nbatches x 1]
        self.wind_pressure = None  # u,v  [Nbatches x 2]
        self.wind = None  # total wind as wind, u, v

        # Histogram --------------
        self.flat_hist = None
        self.hist_vars = None

        # Raw flat values
        self.flat_colors = None  # [Nbatches x N ]
        self.flat_wind = None  # [Nbatches x windchannels ]


    def color_idx_to_center_bary(self, color_idx):
        '''
        Returns u,v barycentric coordinates of patch center for patch with a given idx.
        Only works for color sails.
        :param color_idx: [ N ]
        :return: [ N x 2 ]
        '''
        return self.patch_center_uv[color_idx]


    def color_idx_to_color_bary(self, color_idx):
        print('Patch center uv: %s ' % str(self.patch_center_uv.shape))
        print('Patch uv: %s ' % str(self.patch_uv.shape))
        print('Sh:  %s' % str(self.patch_uv[color_idx].shape) )
        return self.patch_uv[color_idx]


    def compute_bezier_tri_control_points(self, colors, wind):
        '''

        :param colors: [Nbatches x 3 x nchannels(3) ]
        :param wind: [Nbatches x 3] -- all wind params as wind, u, v
        :return: [Nbatches x P x 3] control points.
        '''
        sh = [ tf.shape(wind)[0], 1]
        w = tf.expand_dims(tf.slice(wind, [0,0], sh), axis=2)  # [Nbatches x 1 x 1]
        u = tf.expand_dims(tf.slice(wind, [0,1], sh), axis=2)  # [Nbatches x 1 x 1]
        v = tf.expand_dims(tf.slice(wind, [0,2], sh), axis=2)  # [Nbatches x 1 x 1]
        ones = tf.expand_dims(tf.ones(sh, tf.float32), axis=2)
        zeros = tf.expand_dims(tf.zeros(sh, tf.float32), axis=2)

        # [Nbatches x 10 x 3]
        bez_bary = tf.concat([
            tf.concat([ones, zeros, zeros], axis=2),  # 300  [ Nbatches x 1 x 3 ]
            tf.concat([zeros, ones, zeros], axis=2),  # 030
            tf.concat([zeros, zeros, ones], axis=2),  # 003
            tf.concat([zeros, v, 1 - v], axis=2), # 012
            tf.concat([zeros, u + v, 1 - u - v], axis=2), # 021
            tf.concat([u, zeros, 1 - u], axis=2), # 102
            tf.concat([u + v, zeros, 1 - u - v], axis=2),  # 201
            tf.concat([u, 1 - u, zeros], axis=2), # 120
            tf.concat([1 - v, v, zeros], axis=2), # 210
            tf.concat([u, v, 1 - u - v], axis=2)], # 111
            axis=1)

        # [Nbatches x 1 x 10]
        dist_sq = tf.concat([
            ones * 1000, # 300
            ones * 1000, # 030
            ones * 1000, # 003
            2.0 * u * u, # 012
            2.0 * u * u, # 021
            2.0 * v * v, # 102
            2.0 * v * v, # 201
            2.0 * tf.square(1.0 - u - v), # 120
            2.0 * tf.square(1 - u - v), # 210
            zeros], # 111
            axis=2)

        # BASE CONTROL POINTS ----------
        # P - number of control points in a bezier triangle
        # N - number of vertex colors in palette
        # We want to multiply bez_bary [Nbatches x P x N] * [ Nbatches x N x 3 ]  ->  [Nbatches x P x 3] control points.
        base_points = tf.matmul(bez_bary, colors)

        # BASE WIND VECTOR --------------
        # [Nbatches x 3]
        slice_size = [tf.shape(colors)[0], 1, 3]
        p300 = tf.squeeze(tf.slice(colors, [0, 0, 0], slice_size))
        p030 = tf.squeeze(tf.slice(colors, [0, 1, 0], slice_size))
        p003 = tf.squeeze(tf.slice(colors, [0, 2, 0], slice_size))
        normal = tf.cross(p300 - p030, p300 - p003)  # Intentionally unnormalized # [Nbatches x 3 ]
        constant_wind = normal * tf.squeeze(w, axis=2) * self.opts.max_wind

        # INFLUENCE ----------------------
        # Wind affects different control points differently, depending on the (u,v} positions
        # [Nbatches x 3 x 1] * [Nbatches x 1 x P ]  = [Nbatches x 3 x P]
        pressure = tf.exp(dist_sq * (-1.0 / self.opts.wind_alpha))
        influence = tf.matmul(tf.expand_dims(constant_wind, axis=2), pressure)
        influence = tf.transpose(influence, perm=[0,2,1])

        # FINAL CONTROL POINTS ------------
        control_points = base_points + influence

        return control_points, normal


    def init_deterministic_decoder(self, flat_colors, wind=None, levels=None, soften_alphas=True):
        '''

        :param flat_colors: [Nbatches x N]
        :param wind: either
            [Nbatches x 1] -- just wind
            [Nbatchex x 2 ] -- just pressure
            [Nbatches x 3] -- all wind params as wind, u, v
        :param wchannels: specifies which wind input this is
        :param levels:
        :param soften_alphas:
        :return:
        '''
        self.flat_colors = flat_colors
        self.flat_wind = wind

        with tf.name_scope("palettecolors"):
            # N - number of colors in sparse palette representation
            # Reshape encoding [None N * 3] to be [None N 3] (order does not matter, as it is learned)
            colors_shape = [tf.shape(flat_colors)[0], self.opts.n_colors, self.opts.n_channels]
            LOG.debug('Desired color shape %s' % str(colors_shape))
            self.colors = tf.reshape(flat_colors, tf.stack(colors_shape))

            if soften_alphas and self.opts.n_channels == 4:
                self.colors = tf.add(
                    tf.multiply(self.colors,
                                np.array([1.0, 1.0, 1.0, 0.3], np.float32).reshape([1, 1, 4])),
                    np.array([0.0, 0.0, 0.0, 0.7], np.float32).reshape([1, 1, 4]))

            self.encoding_img = tf.expand_dims(self.colors, 0)

            # Add levels to img
            if levels is not None:
                self.levels = levels
                levels_img = tf.expand_dims(tf.tile(tf.expand_dims(levels, axis=2), [1, 1, 3]), 0)
                self.encoding_img = tf.concat([self.encoding_img, levels_img], axis=2)

            interpolated_vals = self.colors
            num_interpolated_vals = self.opts.n_colors
            interpolator = self.interpolator
            batch_size = tf.shape(flat_colors)[0]
            if self.opts.wind_nchannels > 0 and wind is None:
                raise RuntimeError('Must supply wind if specified in options')

            if wind is not None:
                LOG.info('Training with wind chaneels %d' % self.opts.wind_nchannels)
                if self.opts.wind_nchannels == 1:
                    # Neutral pressure point
                    self.wind_val = wind
                    self.wind_pressure = tf.ones([batch_size, 2], tf.float32) * (1.0 / 3.0)
                    self.wind = tf.concat([self.wind_val, self.wind_pressure], axis=1)
                elif self.opts.wind_nchannels == 2:
                    # Zero wind
                    self.wind_val = tf.zeros([batch_size, 1], tf.float32)
                    self.wind_pressure = wind[..., :2]
                    self.wind = tf.concat([self.wind_val, self.wind_pressure], axis=1)
                elif self.opts.wind_nchannels == 3:
                    # Just use the full wind
                    self.wind = wind[..., :3]
                    self.wind_val = tf.slice(wind, [0,0], [batch_size, 1])
                    self.wind_pressure = tf.slice(wind, [0, 1], [batch_size, 2])
                else:
                    raise RuntimeError('Unsupported number of wind channels %d' % self.opts.wind_nchannels)

                with tf.name_scope("wind_control_points"):
                    interpolated_vals, self.normal = self.compute_bezier_tri_control_points(self.colors, self.wind)
                    self.bern_pts = interpolated_vals
                    num_interpolated_vals = 10
                    interpolator = self.bernstein

                #TODO: add wind to encoding image

            # P - number of colors in rendered palette
            # We want to multiply interpolator [P N] * [N 3] --> [P 3] to get
            # P palette colors for every batch, but matmul does not support this directly,
            # so we do an elaborate reshaping manipulation (see vae_test notebook for sanity checks)
            print('Interpolated Colors shape %s' % str(tf.shape(interpolated_vals)))
            colors_resh = tf.reshape(
                tf.transpose(interpolated_vals, perm=[1, 0, 2]), [num_interpolated_vals, -1])
            LOG.debug(
                'Colors reshape shape %s, interp shape %s' %
                (str(colors_resh.shape), str(interpolator.shape)))

            palette_raw = tf.matmul(interpolator, colors_resh)
            LOG.debug('Result raw palette shape is %s' % str(palette_raw.shape))
            palette_colors = tf.transpose(
                tf.reshape(tf.transpose(palette_raw),
                           [-1, self.opts.n_channels, interpolator.shape[0]]), perm=[0, 2, 1])
            LOG.debug('Final palette colors are %s' % str(palette_colors.shape))
            LOG.debug(palette_colors.graph)

            # self.palette_viz = tf.reshape(
            #     palette_colors,
            #     [-1, self.n_tri, self.interpolator.shape[0] / self.n_tri, self.opts.n_channels])

            self.patch_colors = tf.reshape(
                palette_colors, [-1, self.interpolator.shape[0], self.opts.n_channels])
            LOG.debug('Z shape is %s ' % str(self.patch_colors.shape))

            self.lab_patch_colors = color_ops.rgb2lab_anyshape(self.patch_colors)

            tf.summary.image('train_batch_encoding', color_ops.to_uint8(self.encoding_img), collections=['train'])
            tf.summary.image('test_encoding', color_ops.to_uint8(self.encoding_img), collections=['test'])

            if self.opts.discrete_continuous:
                if levels is None:
                    raise RuntimeError('Must specify levels for discrete-continuous palette')
                self.levels = levels
                self.areas = compute_subdivision_level_areas(
                    levels, self.n_tri, self.opts.max_tri_subdivs)
                patch_areas_aux = tf.multiply(tf.expand_dims(self.areas, axis=1),
                                              tf.expand_dims(self.tiers_mat, axis=0))
                self.patch_areas = tf.reduce_sum(
                    tf.reduce_sum(patch_areas_aux, axis=3), axis=2) * 0.25 * 0.25 * math.sqrt(3.0)


    def init_histogram(self, n_bins, sigma=None):
        if self.flat_hist is not None:
            LOG.warn('Histogram already initialized')
            return

        with tf.name_scope("palette_hist"):
            if sigma is None:
                sigma = 1.0 / n_bins / 2.0
            sigma_sq = sigma * sigma
            self.flat_hist, self.hist_vars = color_ops.compute_rbf_hist(
                tf.expand_dims(self.patch_colors, axis=1), n_bins, sigma_sq, self.patch_areas)


    def visualize_palette(self, img_width, patch_colors, patch_areas=None, levels=None, wind_value=None, wind_location=None, **args):
        patch_levels = None
        if levels is not None:
            patch_levels = levels[self.tri_idx] * self.opts.max_tri_subdivs
        return visualize_palette_cv(img_width, patch_colors, self.verts, self.tiers_idx,
                                    patch_areas=patch_areas, patch_levels=patch_levels,
                                    wind_value=wind_value, wind_location=wind_location, **args)

    def get_viz_for_idx_py_func(self, idx, width):
        def render_closure(idx, width, p):
            return (lambda c:  # just takes in patch_colors
                    p.visualize_palette(width, np.squeeze(c[idx])))

        def render_closure_levels(idx, width, p):
            return (lambda c, a, l:  # takes in patch_colors, areas, levels
                    p.visualize_palette(width, np.squeeze(c[idx]),
                                        patch_areas=a[idx], levels=l[idx]))

        def render_closure_wind(idx, width, p):
            return (lambda c, wv, wl:  # takes in patch_colors, windvalues, windlevels
                    p.visualize_palette(width, np.squeeze(c[idx]),
                                        wind_value=np.squeeze(wv[idx]),
                                        wind_location=np.squeeze(wl[idx])))

        if not self.opts.discrete_continuous:
            if self.opts.wind_nchannels <= 0:
                palette_img = tf.py_func(render_closure(idx, width, self),
                                        [self.patch_colors],
                                        tf.float32)
            else:
                palette_img = tf.py_func(render_closure_wind(idx, width, self),
                                         [self.patch_colors, self.wind_val, self.wind_pressure],
                                         tf.float32)
        else:
            if self.opts.wind_nchannels <= 0:
                palette_img = tf.py_func(render_closure_levels(idx, width, self),
                                        [self.patch_colors, self.patch_areas, self.levels],
                                        tf.float32)
            else:
                raise RuntimeError('Not implemented')

        return palette_img
