import numpy as np

def compute_3d_histogram(img, n_bins):
    '''
    Computes 3D RGB histogram of input image w x h x 3 float32 array, values 0..1.

    Returns:
    idx - list of lists of length 3, denoting RGB bin indexes 0...n_bins-1
    counts - number of elements in each bin in idx
    '''
    bins = np.minimum(
        n_bins-1,
        np.floor(n_bins * np.maximum(0.0, np.minimum(1.0, img)))).astype(np.uint32)
    idx,counts = np.unique(bins.reshape([-1, 3]), axis=0, return_counts=True)
    return (idx, counts)


def histogram_to_3d_array(idx, counts, n_bins):
    '''
    Inflates histogram represented as idx,counts to an n_bins x n_bins x n_bins
    float32 array.

    Returns:
    resulting array
    '''
    res = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    res[idx.T.tolist()] += counts
    return res

def compute_histogram_entropy(hist, eps=1e-6):
    '''
    Takes a n_bins x n_bins x n_bins float32 array and returns a scalar entropy

    Returns:
    single float32 representing entropy
    '''
    s = np.sum(hist)
    assert s != 0, "Found histogram with no entries!"
    
    hist = hist/s 
    #NOTE: normalize by sum

    log_hist = np.log(hist+eps)

    entropy = -1 * np.sum(hist*log_hist)

    return entropy

def write_3d_histogram(idx, counts, n_bins, filename):
    '''
    Writes compact hitogram representation to file.
    '''
    with open(filename, 'w') as f:
        f.write('%d %d %d\n' % (n_bins, n_bins, n_bins))
        for tu in zip(idx, counts):
            f.write('%d %d %d %d\n' % (tu[0][0], tu[0][1], tu[0][2], tu[1]))


def read_3d_histogram(filename, normalize=False):
    '''
    Reads compact histogram representation from file.

    Returns:
    idx - list of lists of length 3, denoting RGB bin indexes 0...n_bins-1
    counts - number of elements in each bin in idx, divided by total if normalize
    n_bins - number of bins per dimension
    '''
    total = 0
    idx = []
    counts = []
    n_bins = None
    with open(filename) as f:
        for line in f:
            elems = line.strip().split(' ')
            coords = [int(x) for x in elems[0:3]]
            if n_bins is None:
                n_bins = coords[0]
                if len([x for x in coords if x != n_bins]) > 0:
                    raise RuntimeError('Variable bin dimensions not supported')
            else:
                count = int(elems[3])
                total += count
                idx.append(coords)
                counts.append(count)

    idx = np.array(idx, dtype=np.uint32)
    counts = np.array(counts, dtype=(np.float32 if normalize else np.uint32))
    if normalize:
        counts /= total
    return (idx, counts, n_bins)


def read_3d_histogram_direct(filename, normalize=False):
    '''
    Reads compact histogram representation from file directly into 3D array.

    Returns:
    3D array with counts, divided by total if normlize==True.
    '''
    res = None
    total = 0
    with open(filename) as f:
        for line in f:
            elems = line.strip().split(' ')
            coords = [int(x) for x in elems[0:3]]
            if res is None:
                res = np.zeros(coords, dtype=np.float32)
            else:
                count = int(elems[3])
                total += count
                res[coords[0], coords[1], coords[2]] = count
    if normalize:
        res = res / (1.0 * total)
    return res
