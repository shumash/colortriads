import os
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize
import random
from learn.tf.log import LOG
import util.hist
import util.io
import util.img_util as img_util
import learn.tf.color.palette as palette


class IndexedFileCacheHelper(object):
    def __init__(self, cache_dir, file_ext, save_function, read_function):
        self.cache_dir = cache_dir
        self.ext = file_ext
        if self.ext[0] != '.':
            self.ext = '.' + self.ext
        self.save_function = save_function
        self.read_function = read_function
        self.hits = [0.0 for i in range(200)]


    def hit_rate(self):
        return sum(self.hits) / len(self.hits)


    def get_cached_filename(self, idx):
        return os.path.join(self.cache_dir, 'line_%07d%s' % (idx, self.ext))


    def get_from_cache(self, idx):
        fname = self.get_cached_filename(idx)
        res = None

        if os.path.isfile(fname):
            # LOG.debug('Getting from cache %s' % fname)
            try:
                res = self.read_function(fname)
            except Exception as e:
                LOG.error('Could not read filename %s with Exception: %s' % (fname, '{0}'.format(e)))

        if res is None:
            self.hits.append(0.0)
        else:
            self.hits.append(1.0)

        self.hits = self.hits[1:len(self.hits)]
        return res


    def save_in_cache(self, idx, obj):
        fname = self.get_cached_filename(idx)
        # LOG.debug('Saving in cache %s' % fname)
        self.save_function(fname, obj)


# FILE LIST DATA HELPER ------------------------------------------------------------------------------------------------
class FileListDataHelper(object):
    def __init__(self, datafile, data_dir, field_num=-1):
        '''

        :param datafile:
        :param data_dir:
        :param field_num: if positive, will interpret that space-delimited field as a file; else will pass whole line downstream
        '''
        if datafile:
            self.datafile = os.path.abspath(datafile)
        else:
            self.datafile = None
        self.data_dir = data_dir
        self.field = field_num
        self.filenames = []
        self.orig_filenames = []
        self.orig_lines = []
        self.__init_filenames()
        self.cache_helper = None


    def add_single_abspath(self, fname, do_checks=True):
        full_fname = FileListDataHelper.__process_filename(fname, self.data_dir, do_checks)
        self.filenames.append(full_fname)
        self.orig_filenames.append(fname)
        self.orig_lines.append('Unknown line')
        

    # Overwrite to e.g. add image size or other settings
    def cache_dir_str(self):
        raise RuntimeError('Caching not supported by this data type')


    def _create_cache_helper(self, cache_dir):
        '''
        Defaults to images; override to create another cache helper.
        :param cache_dir:
        :return:
        '''
        return IndexedFileCacheHelper(cache_dir, '.png',
                                      lambda f, o: imsave(f, o),
                                      lambda f: imread(f).astype(np.float32) / 255.0)


    def enable_file_cache(self, caches_directory):
        '''
        Saves read files to a separate location on disk
        (only makes sense if reading is time-consuming, e.g. if data is generated).

        :param caches_directory: base cache directory; actual cache will be cached_directory/datafile_dir/datafile_basename
        '''
        cache_dir = os.path.join(
            caches_directory, '_'.join(
                ['cache', self.cache_dir_str(), os.path.basename(os.path.dirname(self.datafile)),
                 os.path.basename(self.datafile) ]))
        self.cache_helper = self._create_cache_helper(cache_dir)

        # Create dir if DNE
        if not os.path.isdir(cache_dir):
            LOG.info('Creating cache directory %s' % cache_dir)
            os.makedirs(cache_dir)
        else:
            LOG.info('Cache directory exists %s' % cache_dir)


    @staticmethod
    def read_filenames(datafile, data_dir, field, do_checks=True, return_orig=False, return_lines=False):
        origfilenames = []
        filenames = []
        orig_lines = []
        with open(datafile) as f :
            for line in f :
                parts = [x for x in line.strip().split() if len(x) > 0]
                if field >= 0:
                    if len(parts) > field :
                        fname = parts[field]
                        origfilenames.append(fname)
                        orig_lines.append(line)

                        fname = FileListDataHelper.__process_filename(fname, data_dir, do_checks)
                        filenames.append(fname)
                elif len(parts) > 0:  # Special behavior for negative field
                    filenames.append(' '.join(parts))
            if return_orig and return_lines:
                return filenames, origfilenames, orig_lines
            elif return_orig:
                return filenames, origfilenames
            elif return_lines:
                return filenames, orig_lines
            else:
                return filenames


    @staticmethod
    def __process_filename(fname, data_dir, do_checks):
        res = fname
        if not os.path.isabs(fname):
            res = os.path.join(data_dir, fname)
        if do_checks and not os.path.isfile(res):
            raise RuntimeError('Cannot locate %s' % fname)
        return res


    def __init_filenames(self):
        if self.datafile is not None:
            self.filenames, self.orig_filenames, self.orig_lines = FileListDataHelper.read_filenames(
                self.datafile, self.data_dir, self.field, return_orig=True, return_lines=True)


    def get_specific_batch(self, indexes):
        res = []
        for idx in indexes:
            contents = self._read_file_with_cache(idx)
            if contents is not None:
                res.append(contents)
        if len(res) == 0:
            raise RuntimeError('Could not read any filenames')
        return self._assemble_data(res)


    def get_random_batch(self, count):
        idx = np.random.choice(range(len(self.filenames)), size=(count), replace=False)
        hit_rate = self.cache_helper.hit_rate() if self.cache_helper is not None else 0.0
        LOG.log_every_n(LOG.DEBUG, 'Batch %s (cache hit rate: %0.3f)' % (str(idx), hit_rate), 100)
        return self.get_specific_batch(idx), idx


    def get_all(self):
        return self.get_specific_batch(range(len(self.filenames)))


    def _cache_file(self):
        if self.cache_dir is not None:
            return os.path.join()


    def _read_file_with_cache(self, idx):
        if self.cache_helper is not None:
            contents = self.cache_helper.get_from_cache(idx)
            if contents is not None:
                return contents
        try:
            f = self.filenames[idx]
            contents = self._read_file(f, idx)
        except Exception as e:
            LOG.error('Could not read filename %s with Exception: %s' % (f, '{0}'.format(e)))
            return None

        if self.cache_helper is not None and contents is not None:
            self.cache_helper.save_in_cache(idx, contents)

        return contents


    def _read_file(self, filename, idx):
        raise RuntimeError('__read_file not implemented; must override')


    def _assemble_data(self, read_files):
        raise RuntimeError('__assemble_data not implemented; must override')


# HISTOGRAM DATA HELPER ------------------------------------------------------------------------------------------------

def read_hist_to_row(filename):
    '''
    Reads histogram from file and reshapes to [1 nbins^3]
    '''
    idx,counts,n_bins = util.hist.read_3d_histogram(filename, normalize=True)
    hist3d = util.hist.histogram_to_3d_array(idx,counts,n_bins)
    hist = hist3d.reshape([1, -1])
    return hist


class HistDataHelper(FileListDataHelper):
    def __init__(self, datafile, data_dir, field_num, nbins=10):
        super(HistDataHelper, self).__init__(datafile, data_dir, field_num)
        self.nbins = nbins


    def _read_file(self, filename, idx):
        return read_hist_to_row(filename)


    def _assemble_data(self, read_files):
        count = len(read_files)
        res = np.zeros((count, self.nbins ** 3), dtype=np.float32)
        for row in range(count):
            res[row, :] = read_files[row]
        return res


# IMAGES DATA HELPER ---------------------------------------------------------------------------------------------------

class ImageDataHelper(FileListDataHelper):
    def __init__(self, datafile, data_dir, field_num, img_width=300):
        super(ImageDataHelper, self).__init__(datafile, data_dir, field_num)
        self.width = img_width


    def cache_dir_str(self):
        return 'img_w%d' % self.width


    def _read_file(self, filename, idx):
        return img_util.read_resize_square(filename, self.width)


    def _assemble_data(self, read_files):
        return ImageDataHelper.assemble_image_data(read_files, self.width)


    @staticmethod
    def assemble_image_data(read_files, width):
        count = len(read_files)
        res = np.zeros((count, width, width, 3), dtype=np.float32)
        for row in range(count):
            res[row, :, :, :] = read_files[row]
        return res


class RandomPatchDataHelper(ImageDataHelper):
    '''
    Returns random patches from an image.

    Notes:
        - specify patch_range to cut out variable sized patches, then resize to patch_width
        - specify center_bias to bias patch selection to central pixels
    '''
    def __init__(self, datafile, data_dir, field_num, img_width, patch_width, patch_range=None, center_bias=False):
        super(RandomPatchDataHelper, self).__init__(datafile, data_dir, field_num, img_width)
        self.patch_width = patch_width
        if patch_range is None:
            self.patch_range = (patch_width, patch_width)
        else:
            self.patch_range = patch_range
        self.center_bias = center_bias

    def _assemble_data(self, read_files):
        return ImageDataHelper.assemble_image_data([self.random_patch(x) for x in read_files], self.patch_width)

    def get_random_pos(self, rwidth):
        if not self.center_bias:
            start_row = random.randint(0, self.width - rwidth)
            start_col = random.randint(0, self.width - rwidth)
        else:
            pos = np.random.normal([self.width / 2.0, self.width / 2.0],
                                   [self.width * 0.3, self.width * 0.3]) - rwidth / 2.0
            start_row = int(max(0, min(self.width - rwidth, pos[0])))
            start_col = int(max(0, min(self.width - rwidth, pos[1])))
        return start_row, start_col

    def random_patch(self, img, return_ind=None):
        rwidth = random.randint(self.patch_range[0], self.patch_range[1])
        start_row, start_col = self.get_random_pos(rwidth)

        patch = img[start_row:start_row+rwidth, start_col:start_col+rwidth, :]
        patch = img_util.resize_square_rgb(patch, self.patch_width)
        
        if return_ind:
            return start_col, start_row, rwidth, rwidth, patch
        else:
            return patch

    def random_patch_for_idx(self, idx):
        img = ImageDataHelper._read_file(self, self.filenames[idx], idx)
        return self.random_patch(img, return_ind=True)



class SpecificPatchDataHelper(ImageDataHelper):
    '''
    Reads file of the form (resizes image to img_width; then cuts patch_width at specified location):
    my/file/name.png start_col,start_row,width,height
    OR
    my/file/name.png start_col,start_row  <-- patch_width assumed

    If the file does not contain patch locations, resizes whole image to patch size.

    '''
    def __init__(self, datafile, data_dir, field_num, img_width, patch_width, indexes_field_num=1):
        super(SpecificPatchDataHelper, self).__init__(datafile, data_dir, field_num, img_width)
        self.patch_width = patch_width

        self.patch_indexes = [ [int(x) for x in f.split(',')] for f in
                FileListDataHelper.read_filenames(datafile, '', indexes_field_num, do_checks=False) ]

        if len(self.patch_indexes) > 0 and (len(self.patch_indexes) != len(self.filenames)):
            raise RuntimeError('Unequal filenames/patch index sizes %d vs %d' %
                               (len(self.filenames), len(self.patch_indexes)))

    def _read_file(self, filename, idx):
        if len(self.patch_indexes) > 0:
            img = img_util.read_resize_square(filename, self.width)
            sidx = self.patch_indexes[idx]
            pwidth = self.patch_width
            if len(sidx) > 2:
                pwidth = sidx[2]
            patch = img[sidx[1]:sidx[1]+pwidth, sidx[0]:sidx[0]+pwidth]
            return img_util.resize_square_rgb(patch, self.patch_width)
        else:
            img = img_util.read_resize_square(filename, self.patch_width)
            return img

    def _assemble_data(self, read_files):
        return ImageDataHelper.assemble_image_data(read_files, self.patch_width)


class ColorGlimpsesDataHelper():
    def __init__(self, num, width):
        self.num = num
        self.width = width

    def get_random_batch(self, count):
        res = np.zeros([count, self.width, self.width], np.float32)

        for b in range(count):
            for i in range(self.num):
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.width-1)
                res[b, x, y] = 1.0
        return res


class ComputedHistDataHelper(FileListDataHelper):
    def __init__(self, datafile, data_dir, img_field_num, hist_field_num, hist_save_dir, nbins=10, img_width=300):
        super(ComputedHistDataHelper, self).__init__(datafile, data_dir, img_field_num)
        self.image_helper = ImageDataHelper(None, '', 0, img_width=img_width)
        self.hist_helper = HistDataHelper(None, '', 0, nbins=nbins)
        self.hist_filenames = {}
        self.hist_save_dir = hist_save_dir

        LOG.info('Saving computed histograms in ComputedHistDataHelper to %s' % hist_save_dir)
        hf = [os.path.join(self.hist_save_dir, os.path.basename(f)) for f in
              FileListDataHelper.read_filenames(datafile, '', hist_field_num, do_checks=False)]
        self.hist_filenames = dict(zip(self.filenames, hf))

    # Add caching support
    # def cache_dir_str(self):
    #     return 'img%d' % self.width

    def _read_file(self, filename, idx):
        if filename not in self.hist_filenames:
            print(self.hist_filenames)
            raise RuntimeError('Histogram file not found %s' % filename)
        hfilename = self.hist_filenames[filename]
        if not os.path.isfile(hfilename) :
            LOG.debug('Computing histogram for image %s ' % filename)
            img = self.image_helper._read_file(filename, idx)
            imsave(hfilename + '.png', img)
            (idx, counts) = util.hist.compute_3d_histogram(img, self.hist_helper.nbins)
            util.hist.write_3d_histogram(idx, counts, self.hist_helper.nbins, hfilename)
        return self.hist_helper._read_file(hfilename, idx)


    def _assemble_data(self, read_files):
        return self.hist_helper._assemble_data(read_files)


# EXTERNAL PALETTE DATA HELPER -----------------------------------------------------------------------------------------

class ExternalPaletteDataHelper(FileListDataHelper):
    '''
    Reads colors from an external file and renders them into an image (base classes determine that).
    E.g. look at data/color/themes.
    '''
    def __init__(self, datafile, img_width=300, max_tri_subdivs=5):
        super(ExternalPaletteDataHelper, self).__init__(datafile, data_dir='')
        self.width = img_width
        self.max_subdivs = max_tri_subdivs
        self._init_random_params()


    def cache_dir_str(self):
        return 'synthhex_w%d_sub%d' % (self.width, self.max_subdivs)


    def _render_palette(self, palette_info):
        res, verts, n_tri, tiers_mat, tiers_idx, tri_idx = \
            palette.create_rgb_tri_interpolator(palette_info['colors'].shape[0], self.max_subdivs)
        tri_colors = res.dot(palette_info['colors'])
        patch_levels = palette_info['levels'][tri_idx]
        img = palette.visualize_palette_cv(self.width, tri_colors, verts, tiers_idx, None, None, patch_levels,
                                           skip_rendering_higher_levels=True)
        return img


    def _init_random_params(self):
        for i in range(len(self.filenames)):
            f = self.filenames[i]
            n_vert = len(f.strip().split())
            n_tri = len(palette.get_activated_triangles(n_vert))
            levels = [random.randint(0, self.max_subdivs) for x in range(n_tri)]
            self.filenames[i] = f + '|' + ','.join([str(x) for x in levels])


    def _assemble_data(self, read_files):
        return ImageDataHelper.assemble_image_data(read_files, self.width)


    def _read_file(self, line, idx):
        parts0 = line.strip().split('|')
        if len(parts0) != 3:
            raise RuntimeError('Could not read line: %s' % line)

        palette_info = { 'url' : parts0[0] }
        cparts = [p for p in parts0[1].split() if len(p) > 0]

        colors = []
        for i in range(len(cparts)):
            p = cparts[i]
            comps = [ int(x) for x in p.split(',') ]
            if len(comps) != 4:
                raise RuntimeError('Malformed external palette line: %s' % line)
            colors.append([comps[3], comps[0], comps[1], comps[2]]) # percent,R,G,B
            colors.sort(reverse=True)

        levels = [int(l) for l in parts0[2].split(',') if len(l) > 0]
        palette_info['colors'] = np.array([ [c[1]/255.0, c[2]/255.0, c[3]/255.0] for c in colors ], dtype=np.float32)  # R,G,B,percent
        palette_info['sizes'] = [ c[0] for c in colors ]
        palette_info['levels'] = np.array(levels, dtype=np.uint32)
        return self._render_palette(palette_info)


# Data randomization ---------------------------------------------------------------------------------------------------

class RandomGrayscaler():
    def __init__(self, img_width, random_bw_fraction=-1, color_glimpses=-1):
        self.bw_fraction = random_bw_fraction
        self.glimpses_synthesizer = None
        self.nchannels = 3
        if self.bw_fraction > 0:
            if self.bw_fraction > 1.0:
                LOG.warning('All input is Black and White with bw_fraction set > 1.0')
            if color_glimpses > 0:
                self.nchannels = 5  # Bw, R, G, B, mask
                raise RuntimeError('Glimpses not implemented; need to think through nchannels')
                self.glimpses_synthesizer = ColorGlimpsesDataHelper(color_glimpses, width=img_width)


    def process_batch(self, imgs, do_mask=True):
        if self.bw_fraction <= 0.0:
            return imgs

        batchsize = imgs.shape[0]
        res = np.zeros([batchsize, imgs.shape[1], imgs.shape[2], self.nchannels], imgs.dtype)
        for b in range(batchsize):
            use_bw = do_mask and (random.random() < self.bw_fraction)

            if self.glimpses_synthesizer is None:
                # Input is either RGB, or GrayGrayGray
                if use_bw:
                    bw = np.sum(imgs[b, :, :, :], axis=2) / 3.0
                    for i in range(self.nchannels):
                        res[b, :, :, i] = bw
                else:
                    res[b, :, :, :] = imgs[b, :, :, :] # pass along unchanged
            else:
                # Input is GrayRGBA (somewhat redundant; could kill one dimension, but we avoid LAB for now)
                bw = np.sum(imgs[b, :, :, :], axis=3) / 3.0  # convert to grayscale

                res[b, :, :, 0] = bw
                if use_bw:
                    glimpses = np.squeeze(self.glimpses_synthesizer.get_random_batch(1))
                else:
                    glimpses = np.ones([imgs.shape[1], imgs.shape[2]], np.float32)
                for i in range(3):
                    res[b, :, :, i+1] = imgs[b, :, :, i] * glimpses
                res[b, :, :, 4] = glimpses
        return res
