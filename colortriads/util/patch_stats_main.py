import os
import argparse
import numpy as np
import cPickle as pkl
import tqdm
from multiprocessing import Pool
from functools import partial
import skimage.color as color

import util.hist
import util.img_util

# NOTE: To use the image as a single patch,
# make IMG_WIDTH and PATCH_WIDTH the same
IMG_WIDTH = 512
PATCH_WIDTH = 32
N_BINS = 10
N_WORKERS = 32
CHUNK_SIZE = 10000

def arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str,
        default='/ais/gobi5/amlan/color/')
    parser.add_argument('--file_list', type=str,
        default='/h/14/amlan/gitrepo/color_scapes/experiments/color/data/all_imgs/keys_all.txt')
    parser.add_argument('--out_file', type=str,
        default='/ais/gobi5/amlan/color/data_stats.pkl')

    args = parser.parse_args()

    return args

def process_img(rel_path, data_dir):
    img_path = os.path.join(data_dir, rel_path)
    try:
        img = util.img_util.read_resize_square(img_path, IMG_WIDTH)
    except:
        print('Error in reading %s'%(rel_path))
        return (rel_path, {})

    es = {}
    es['entropy'] = {}
    es['mean'] = {}
    es['hsv'] = {}

    for py in range(0, IMG_WIDTH, PATCH_WIDTH):
        for px in range(0, IMG_WIDTH, PATCH_WIDTH):
            # NOTE: no overlap in patches!
            tmp_img = img[py:py+PATCH_WIDTH, px:px+PATCH_WIDTH, :]
            es['mean'][(px, py)] = np.mean(tmp_img, axis=(0,1))
 
            tmp_img_hsv = color.rgb2hsv(tmp_img)    
            es['hsv'][(px, py)] = np.mean(tmp_img_hsv, axis=(0,1))

            idx, counts = util.hist.compute_3d_histogram(tmp_img, N_BINS)
            hist = util.hist.histogram_to_3d_array(idx, counts, N_BINS)
        
            e = util.hist.compute_histogram_entropy(hist)
            es['entropy'][(px, py)] = e

    return (rel_path, es)

if __name__ == '__main__':
    args = arguments()
    with open(args.file_list, 'r') as f:
        paths = f.read().strip().split('\n')

    write = True
    if os.path.isfile(args.out_file + '_0'):
        x = raw_input('%s already exists! Overwrite? [y/n]'%(args.out_file + '_0'))
        if x != 'y' and x != 'Y':
            write = False

    if not write:
        import sys
        sys.exit(0)

    num_chunks = 1 + len(paths)//CHUNK_SIZE

    p = Pool(N_WORKERS)
    for i in range(num_chunks):
        tmp_paths = paths[i * CHUNK_SIZE: min((i+1)*CHUNK_SIZE, len(paths))]
        result = list(tqdm.tqdm(p.imap(partial(process_img, data_dir=args.data_dir), tmp_paths), 
                      total=len(tmp_paths), desc='Chunk%d'%(i)))

        print('Writing to file %s'%(args.out_file + '_' + str(i)))
        pkl.dump(result, open(args.out_file + '_' + str(i), 'w'))    
    
    p.close()
