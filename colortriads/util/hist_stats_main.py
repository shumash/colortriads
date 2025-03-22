import matplotlib
matplotlib.use('Agg')

import numpy as np
import cPickle as pkl
import tqdm

from glob import glob
from matplotlib import pyplot as plt
from collections import defaultdict

FILE_P = '/ais/gobi5/amlan/color/stats/data_stats.pkl'

FILES = glob(FILE_P + '*') #Get all files with this prefix

ENTROPIES = defaultdict(list)

# Define classes for the histogram
CLASSES = { 
    'digital_painting':'Art',
    'illustration':'Art',
    'painting':'Art',
    'graphic_design':'Graphic Design',
    'viz':'Viz',
    'photo':'Photo'
}

COLORS = ['orange', 'blue', 'red', 'green']

for i,f in enumerate(FILES):
    data = pkl.load(open(f,'r'))
    for im in tqdm.tqdm(data, desc='Chunk%d'%(i)):
        name = im[0]
        try:
            class_name = CLASSES[name.split('/')[0]]
        except:
            continue

        d = im[1]

        try:
            # Some of the files had errors while processing
            # res = np.mean(d['entropy'].values())
            # res = np.median(d['entropy'].values())
            # Image wise mean or median
            
            res = d['entropy'].values() #Patch wise

        except:
            continue

        ENTROPIES[class_name].extend(res)

n_classes = len(ENTROPIES.keys())
fig, ax = plt.subplots(n_classes, 1, squeeze=True)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)

for i,c in enumerate(ENTROPIES.keys()):
    a = ax[i]
    es = ENTROPIES[c]
    n, bins, patches = a.hist(es, 50, range=(0,5),
        alpha=0.7, label=c, color=COLORS[i])
    a.legend(loc='upper right')
    a.grid(which='both')

plt.xlabel('Entropy')
plt.ylabel('Frequency', labelpad=20)
plt.title('Histogram of mean patch entropy of colours')
fig.tight_layout()

plt.savefig('/ais/gobi5/amlan/color/stats/separate_hist_stats.pdf')
