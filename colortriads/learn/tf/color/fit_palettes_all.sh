#!/bin/bash -e

set -o nounset

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR/../../..

IMGDIR=$1
OUTDIR=$2
mkdir -p $OUTDIR


CHECKPOINT=$SCRIPT_DIR/../../../../models/out/

for I in $(find $IMGDIR -type f -name '*.png' -o -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg"); do
    bname=$(basename $I)
    name="${bname%.*}"

    echo "Processing $name"
    ls -lh $I

    W=$(magick identify -format "%w" $I)
    IMG_WIDTH=512
    if [ "$W" -lt $IMG_WIDTH ]; then
        IMG_WIDTH=$W
    fi

    python -m learn.tf.color.fit_palette_main \
           --max_tri_subdivs=16 \
           --encoder_mode=0 \
           --palette_graph_mode=0 \
           --wind_num_channels=3 \
           --img_width=$IMG_WIDTH \
           --max_colors=3 \
           --palette_layer_specs='700;200;50' \
           --conv_filter_sizes=4,4,4 \
           --conv_filter_counts=64,128,256 \
           --data_input_field=0 \
           --input_mode=2 \
           --patch_width=32 \
           --losses=L2RGB \
           --n_hist_bins=10 \
           --palette_graph_mode=0 \
           --loss_weights=1.0 \
           --restore_palette_graph=$CHECKPOINT \
           --input=$I \
           --output_prefix=$OUTDIR/${name}
done
