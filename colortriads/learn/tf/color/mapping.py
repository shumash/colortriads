
import math
import numpy as np

from util.io import timed_print
from scipy.spatial import distance

from skimage import color

# During training:
# Sail colors = Run(input)
# Mapping = make_mapping(Sail colors, img)
# Loss = colors - sail_colors(mapping)


class Mapping(object):
    pass


def log_tensor(t, name):
    print('{}: {} {}'.format(name, t.dtype, t.shape))


def hash_color(nsub, uint_color):
    uint_color = np.squeeze(uint_color)
    bucket_width = math.ceil(255.0 / nsub)
    nbits = int(math.ceil(math.log(nsub, 2)))
    bucket_width = int(bucket_width)
    if nbits * 3 >= 32:
        raise RuntimeError('Cannot encode color with %d bits per channel' % nbits)
    val = (uint_color[..., 0] / bucket_width).astype(np.uint32)
    val = (val << int(nbits)) + uint_color[..., 1] / bucket_width
    val = val.astype(np.uint32)
    val = (val << int(nbits)) + uint_color[..., 2] / bucket_width
    return val


def unhash_color(nsub, hval):
    if np.ndim(hval) == 0:
        hval = np.array([hval], dtype=np.uint32)
    bucket_width = math.ceil(255.0 / nsub)
    nbits = int(math.ceil(math.log(nsub, 2)))
    bucket_width = int(bucket_width)
    hval = hval.astype(np.uint32)
    red = (hval >> (nbits * 2))
    green = (hval >> nbits) - (red << nbits)
    blue = hval - (red << (nbits * 2)) - (green << nbits)
    return np.concatenate([np.expand_dims(red * bucket_width, axis=1),
                           np.expand_dims(green * bucket_width, axis=1),
                           np.expand_dims(blue * bucket_width, axis=1)], axis=1).astype(np.uint8)


class ColorLookup(object):

    def __init__(self, nsub, colors):
        '''
        :param nsub: number of bins per side of the hash
        :param colors: np array of shape N x 3 and type uint8
        '''
        self.nsub = nsub
        # hash : M x 3 array of unique colors in that cell
        self.cells = {}
        # hash: array of color counts in same order as above
        self.cell_color_counts = {}
        # hash: array of metadata in same order as above
        self.cell_color_idx = {}
        # H x 3 np array of cell center colors for fast search
        self.cell_centers = None
        # H x 1 np vector of hash ids in same order as cell_centers
        self.cell_ids = None

        timed_print('ColorLookup sub %d' % nsub)

        cell_width = 255 / nsub
        unique_colors, unique_indices, unique_counts = np.unique(colors, axis=0, return_index=True, return_counts=True)

        timed_print('ColorLookup sub %d - got unique colors' % nsub)

        hashes = hash_color(nsub, unique_colors)
        for i in range(unique_colors.shape[0]):
            color = unique_colors[i,:]
            h = hashes[i]
            if h not in self.cells:
                self.cells[h] = []
                self.cell_color_counts[h] = []
            self.cells[h].append(np.expand_dims(color, axis=0))
            self.cell_color_counts[h].append(unique_counts[i])

        timed_print('ColorLookup sub %d - got hashes' % nsub)

        self.cell_centers = np.zeros((len(self.cells.keys()), 3), dtype=np.uint8)
        self.cell_ids = self.cells.keys()
        for i in range(len(self.cell_ids)):
            h = self.cell_ids[i]
            color = unhash_color(self.nsub, h)
            self.cells[h] = np.concatenate(self.cells[h], axis=0).astype(np.float16)
            self.cell_centers[i, :] = (color.astype(np.float16) + cell_width / 2)

        timed_print('ColorLookup sub %d - got index' % nsub)


    def print_all(self):
        '''
        Prints contents of the lookup for debug purposes.
        '''
        for h in self.cells.keys():
            content = self.cells[h]
            counts = self.cell_color_counts[h]
            idx = self.cell_ids.index(h)
            center = self.cell_centers[idx]
            print('Cell %d (%d), center (%0.2f %0.2f %0.2f):' % (h, content.shape[0], center[0], center[1], center[2]))
            print(content.shape)
            for c in range(content.shape[0]):
                print('   %d %d %d (count %d)' % (content[c, 0], content[c, 1], content[c, 2], counts[c]))


    def cell_id(self, color):
        return hash_color(self.nsub, color)


    def cell_center(self, cell_id):
        idx = self.cell_ids.index(cell_id)
        return self.cell_centers[idx]


    def has_cell(self, cell_id):
        return cell_id in self.cells


    def nearest_cell(self, color):
        '''
        Returns nearest cell to to the color. If a cell to which a color hashes exists, that
        cell is returned. Else, the cell with its center the closest to color is returned.
        '''
        color_hash = hash_color(self.nsub, color)
        if color_hash in self.cells:
            return color_hash

        distances = np.linalg.norm(self.cell_centers - color.astype(np.float16), axis=1)
        min_idx = np.argmin(distances)
        return self.cell_ids[min_idx]


    def nearest_cell_vec(self, colors, color_hashes=None):
        '''
        Same as nearest_cell, but vectorized.
        '''
        if color_hashes is None:
            color_hashes = hash_color(self.nsub, colors)

        nearest_cells = np.array([ h if h in self.cells else 0 for h in color_hashes], dtype=np.uint32)
        unmatched = np.where(nearest_cells == 0)

        distances = distance.cdist(colors[unmatched].astype(np.float16), self.cell_centers)
        min_idx = np.argmin(distances, axis=1)

        nearest_cells[unmatched] = [self.cell_ids[x] for x in min_idx]
        return nearest_cells


    def nearest_color_in_cell(self, color, cell_hash):
        '''
        Returns the exact color within cell_hash that is closest to color by L2 metric as well as its count.
        :return: (color, count)
        '''
        cell_colors = self.cells[cell_hash]
        counts = self.cell_color_counts[cell_hash]

        distances = np.linalg.norm(cell_colors - color, axis=1)
        min_idx = np.argmin(distances)
        return (cell_colors[min_idx, :].astype(np.uint8), counts[min_idx])


    def nearest_colors_in_cell_vec(self, colors, cell_hash):
        '''
        Same as nearest_color_in_cell, but vectorized.
        :return: (colors, counts)
        '''
        cell_colors = self.cells[cell_hash]
        #print('Query colors %s, source colors %s' % (str(colors.shape), str(cell_colors.shape)))
        counts = self.cell_color_counts[cell_hash]
        distances = distance.cdist(colors.astype(np.float16), cell_colors)
        min_idx = np.argmin(distances, axis=1)
        nearest_colors = cell_colors[min_idx]
        nearest_counts = [counts[x] for x in min_idx]
        return nearest_colors.astype(np.uint8), nearest_counts


    def nearest_color_approx(self, uint_color):
        '''
        Finds approximate nearest color, within error sqrt(3*(255/nsub)^2) if the actual nearest color is in the
        adjacent cell.
        :param uint_color:
        :return: (uint8 color, color metadata)
        '''
        cell_idx = self.nearest_cell(uint_color)
        return self.nearest_color_in_cell(uint_color, cell_idx)


def make_exact_mapping_unoptimized(colors, image):
    float_colors = colors.astype(np.float16)
    imgcolors = image[:, :, 0:3].reshape(-1, 3).astype(np.float16)
    recon = np.zeros((image.shape[0] * image.shape[1], 3), dtype=np.uint8)

    for i in range(image.shape[0] * image.shape[1]):
        color = imgcolors[i, :]
        distances = np.linalg.norm(float_colors - color, axis=1)
        min_idx = np.argmin(distances)
        recon[i, :] = colors[min_idx, :]

    recon_img = recon.reshape(image.shape[0], image.shape[1], 3)
    return recon_img


def make_exact_mapping_unoptimized2(colors, image):
    float_colors = colors.astype(np.float16)
    imgcolors = image[:, :, 0:3].reshape(-1, 3).astype(np.float16)

    distances = distance.cdist(imgcolors, float_colors)
    min_idx = np.argmin(distances, axis=1)
    recon = colors[min_idx, :]

    recon_img = recon.reshape(image.shape[0], image.shape[1], 3)
    return recon_img




# TODO: this one can be much faster; delete or optimize if needed
# def make_approx_mapping_rough(colors, image, nsub=10, source_h=None, target_h=None):
#     imgcolors = image[:,:,0:3].reshape(-1, 3)
#
#     if source_h is None:
#         source_h = ColorLookup(nsub, colors)
#
#     if target_h is None:
#         target_h = ColorLookup(nsub, imgcolors)
#
#     colormap = {}
#     for h in target_h.cells.keys():
#         nearest = source_h.nearest_cell(target_h.cell_center(h))
#         colormap[h] = source_h.cell_center(nearest)
#
#     recon = np.zeros((image.shape[0] * image.shape[1], 3), dtype=np.uint8)
#     hashes = hash_color(nsub, imgcolors)
#     for i in range(image.shape[0] * image.shape[1]):
#         recon[i,:] = colormap[hashes[i]]
#
#     recon_img = recon.reshape(image.shape[0], image.shape[1], 3)
#     return recon_img


def _make_approx_mapping_unvectorized(colors, image, nsub=10, source_h=None):
    '''
    Same as make_approx_mapping, but slow, -- written as a sanity check.
    '''
    imgcolors = image[:, :, 0:3].reshape(-1, 3)
    hashes = hash_color(255, imgcolors)
    unique_hashes = np.unique(hashes)
    unique_colors = unhash_color(255, unique_hashes)
    print('Unique hashes %s' % str(unique_hashes.shape))

    if source_h is None:
        source_h = ColorLookup(nsub, colors)

    colormap = {}
    for i in range(unique_hashes.size):
        h = unique_hashes[i]
        color = unique_colors[i,:]
        nearest_color, count = source_h.nearest_color_approx(color)
        colormap[h] = nearest_color

    timed_print('Made colormap')

    recon = np.zeros((image.shape[0] * image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0] * image.shape[1]):
        recon[i, :] = colormap[hashes[i]]

    recon_img = recon.reshape(image.shape[0], image.shape[1], 3)
    return recon_img


def make_exact_mapping(colors, image, return_idx=False, use_lab=False):
    ncolors = colors.shape[0]
    hashes = hash_color(255, colors)
    unique_hashes, color_idx = np.unique(hashes, return_index=True)
    colors = unhash_color(255, unique_hashes)
    print('%d source colors, %d unique' % (ncolors, colors.shape[0]))

    imgcolors = image[:, :, 0:3].reshape(-1, 3)
    hashes = hash_color(255, imgcolors)
    unique_hashes, hash_idx = np.unique(hashes, return_inverse=True)
    unique_colors = unhash_color(255, unique_hashes).astype(np.float32)
    print('%d image colors, %d unique' % (imgcolors.shape[0], unique_colors.shape[0]))

    src_colors = colors
    if use_lab:
        colors = np.squeeze(color.rgb2lab(np.expand_dims(colors, axis=0)/255.0))
        unique_colors = np.squeeze(color.rgb2lab(np.expand_dims(unique_colors, axis=0)/255.0))

    distances = distance.cdist(unique_colors, colors.astype(np.float32))
    min_idx = np.argmin(distances, axis=1)
    timed_print('Got min distances')

    recon = np.zeros((image.shape[0] * image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0] * image.shape[1]):
        recon[i, :] = src_colors[min_idx[hash_idx[i]]]
    recon_img = recon.reshape(image.shape[0], image.shape[1], 3)

    if not return_idx:
        return recon_img

    ids_img = np.zeros((image.shape[0] * image.shape[1]), dtype=np.int64)
    for i in range(image.shape[0] * image.shape[1]):
        ids_img[i] = color_idx[min_idx[hash_idx[i]]]
    ids_img = ids_img.reshape(image.shape[0], image.shape[1])

    return recon_img, ids_img


def make_approx_mapping(colors, image, nsub=10, source_h=None):
    '''
    Maps each color in image to the (approximately) clostest color in colors, with
    the maximum error of sqrt(3*(255/nsub)^2) in RGB per color.

    :param colors: ncolors x 3 uint8 array
    :param image: w x h x 3 uint8 array
    :param nsub: integer number of subdivisions to use for color lookup
    :param source_h: if exists, assumes this is ColorLookup(nsub, colors)
    :return: reconstructed image
    '''
    imgcolors = image[:, :, 0:3].reshape(-1, 3)
    hashes = hash_color(255, imgcolors)
    unique_hashes, hash_idx = np.unique(hashes, return_inverse=True)
    unique_colors = unhash_color(255, unique_hashes)
    print('Unique hashes %s' % str(unique_hashes.shape))

    if source_h is None:
        source_h = ColorLookup(nsub, colors)

    nearest_cells = source_h.nearest_cell_vec(unique_colors)
    unique_cells, cell_idx = np.unique(nearest_cells, return_inverse=True)

    colormap = {}
    for cidx in range(len(unique_cells)):
        cell = unique_cells[cidx]
        mapped_idx = np.where(cell_idx == cidx)[0]
        mapped_colors = unique_colors[mapped_idx]
        nearest_colors, nearest_counts = source_h.nearest_colors_in_cell_vec(mapped_colors, cell)

        for i in range(mapped_colors.shape[0]):
            nearest_color = nearest_colors[i]
            ucolor_idx = mapped_idx[i]
            uhash = unique_hashes[ucolor_idx]
            colormap[uhash] = nearest_color

    recon = np.zeros((image.shape[0] * image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0] * image.shape[1]):
        recon[i, :] = colormap[hashes[i]]

    recon_img = recon.reshape(image.shape[0], image.shape[1], 3)
    return recon_img



def make_smooth_mapping(tbd):
    # Apply TV regularizer
    pass