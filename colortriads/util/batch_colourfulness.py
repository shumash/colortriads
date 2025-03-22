from util.img_util import colourfulness
import skimage.io as sio
import tqdm
from multiprocessing import Pool
from functools import partial

N_WORKERS = 32
DATA_HOME = '/ais/gobi5/shumash/Data/Color/images512/'
FILE_LIST = '/ais/gobi5/shumash/Data/Color/splits/splits512/total/ALL.txt'
OUT_FILE_LIST = '/ais/gobi5/shumash/Data/Color/splits/splits512/total/ALL_color.txt'

def process_img(fname, data_dir):
    fn = data_dir + fname
    try:
        img = sio.imread(fn)
    except:
        return -1

    return colourfulness(img)

with open(FILE_LIST, 'r') as f:
    fnames = f.read().strip().split('\n')

p = Pool(N_WORKERS)
result = list(tqdm.tqdm(p.imap(partial(process_img, data_dir=DATA_HOME), fnames), 
    total=len(fnames)))

with open(OUT_FILE_LIST, 'w') as f:
    for i in range(len(fnames)):
        f.write('%s %.2f\n'%(fnames[i], result[i]))

