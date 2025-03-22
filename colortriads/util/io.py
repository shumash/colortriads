import numpy as np
from skimage.io import imread
import time, datetime

def timed_print(msg):
    ts = time.time()
    print('%s %s' % (datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f'), msg))

def read_float_img(filename):
    return imread(filename).astype(np.float32) / 255.0

def read_uint_img(filename):
    return imread(filename).astype(np.uint8)

def read_uint_colors_ascii(filename):
    colors = []
    with open(filename) as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) > 0:
                r = int(elems[0])
                g = int(elems[1])
                b = int(elems[2])
                colors.append(np.expand_dims(np.array([r, g, b], dtype=np.uint8), axis=0))
    return np.concatenate(colors, axis=0)

def write_uint_colors_ascii(colors, filename):
    with open(filename, 'w') as f:
        for r in range(colors.shape[0]):
            f.write('%d %d %d\n' % (colors[r, 0], colors[r, 1], colors[r, 2]))

def read_flow(fname):
    """
    Reads .flo flow from filename string, and returns:

    """
    # Sourced from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!

    with open(fname, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print('Reading %d x %d flo file' % (w, h))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            #print data.shape
            return np.reshape(data, (int(h), int(w), 2))
