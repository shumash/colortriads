import argparse
import numpy as np
from skimage.io import imsave

import util.io
import learn.tf.color.mapping as mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get color mapping')
    parser.add_argument(
        '--source_colors', action='store', type=str, default='')
    parser.add_argument(
        '--source_image', action='store', type=str, default='')
    parser.add_argument(
        '--target_colors', action='store', type=str, default='')
    parser.add_argument(
        '--target_image', action='store', type=str, default='')
    parser.add_argument(
        '--output_image', action='store', type=str, default='')
    parser.add_argument(
        '--output_colors', action='store', type=str, default='')
    parser.add_argument(
        '--output_stats', action='store', type=str, default='')
    parser.add_argument(
        '--exact', action='store_true', default=False)
    args = parser.parse_args()


    source_colors = None
    if len(args.source_colors) > 0:
        source_colors = util.io.read_uint_colors_ascii(args.source_colors)
        if len(args.source_image) > 0:
            raise RuntimeError('Only one of --source_colors or --source_image should be set')
    elif len(args.source_image) > 0:
        source = util.io.read_uint_img(args.source_image)
        source_colors = source[:, :, 0:3].reshape(-1, 3)
    else:
        raise RuntimeError('Must set --source_colors or --source_image')

    target = None
    if len(args.target_colors) > 0:
        target_colors = util.io.read_uint_colors_ascii(args.target_colors)
        target = np.expand_dims(target_colors, axis=0)
        if len(args.target_image) > 0:
            raise RuntimeError('Only one of --target_colors or --target_image should be set')
    elif len(args.target_image) > 0:
        target = util.io.read_uint_img(args.target_image)
        target_colors = target[:, :, 0:3].reshape(-1, 3)
    else:
        raise RuntimeError('Must set --target_colors or --target_image')

    util.io.timed_print('Start Mapping')
    if args.exact:
        recon = mapping.make_exact_mapping(source_colors, target)
    else:
        recon = mapping.make_approx_mapping(source_colors, target)
    recon_colors = recon.reshape(-1, 3)
    util.io.timed_print('End Mapping')

    if len(args.output_image) > 0:
        imsave(args.output_image, recon)

    if len(args.output_colors) > 0:
        util.io.write_uint_colors_ascii(recon_colors, args.output_colors)

    if len(args.output_stats) > 0:
        distances = np.linalg.norm(target_colors.astype(np.float64) - recon_colors.astype(np.float64), axis=1)
        mean_l2 = np.sum(distances) / distances.size
        std = np.std(distances)
        with open(args.output_stats, 'w') as f:
            f.write('%0.5f %0.5f\n' % (mean_l2, std))





