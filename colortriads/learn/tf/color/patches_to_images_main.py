import os
import re
import argparse
import numpy as np
from skimage.io import imsave

import util.io
import learn.tf.color.data


def log_tensor(t, name):
    print('{}: {} {}'.format(name, t.dtype, t.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get color mapping')
    parser.add_argument(
        '--base_image_dir', action='store', type=str, default='')
    parser.add_argument(
        '--patch_list_files', action='store', type=str, required=True,
        help='CSV list of patch files.')
    parser.add_argument(
        '--patch_output_dir', action='store', type=str, required=True)
    parser.add_argument(
        '--output_file', action='store', type=str, required=True)
    args = parser.parse_args()

    #print('Making dir: %s' % args.patch_output_dir)
    #os.makedirs(args.patch_output_dir, exist_ok=True)

    outf = open(args.output_file, 'w')
    fnames = args.patch_list_files.split(',')
    global_idx = 0
    for fname in fnames:
        phelper = learn.tf.color.data.SpecificPatchDataHelper(
            fname, args.base_image_dir, field_num=0,
            img_width=512, patch_width=32)

        # HACK-ish (find filename annotation)
        filetag = ''
        m = re.match('.*(medium|hard|easy).*', os.path.basename(fname))
        if m is not None:
            filetag = '%s.' % m.groups()[0]
        if len(filetag) == 0:
            raise RuntimeError('File %s did not have medium/hard/easy tag' % fname)
        # END OF HACK-ish

        print('Read {} patch specs from {}'.format(len(phelper.filenames), fname))
        for idx in range(len(phelper.filenames)):

            # HACK (specific to WEB patches)
            img_src = phelper.filenames[idx]
            tag = re.sub(".*WEB/", "", img_src).split("/")[0]
            # END OF HACK

            pdata = phelper.get_specific_batch([idx])
            patch = (pdata[0,...] * 255).astype(np.uint8)
            patch_name = '%s%s.patch%06d.png' % (filetag, tag, global_idx)
            pout_file = os.path.join(args.patch_output_dir, patch_name)
            #log_tensor(pdata, 'pdata')
            #log_tensor(patch, 'patch')

            imsave(pout_file, patch)
            outf.write('%s %s\n' % (patch_name, phelper.orig_lines[idx].strip()))

            # print('Processing patch {} : {}'.format(patch_name, img_src))
            global_idx += 1
        print('Processed %d' % global_idx)