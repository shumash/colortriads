#!/bin/bash -e

set -o nounset

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR/../../..

IMGDIR=$1
PALETTEDIR=$2
OUTDIR=$3
mkdir -p $OUTDIR


for I in $(ls $IMGDIR/*.{jpg,jpeg,JPG,png,PNG,bmp,BMP} 2>/dev/null || true | grep -v "_alpha.png"); do
    bname=$(basename $I)
    name="${bname%.*}"

    echo "Processing $name"
    ls -lh $I

    PAL=$PALETTEDIR/${name}.palette.txt
    if [ ! -f $PAL ]; then
        echo "Palette $PAL does not exist"
        continue
    fi
    ls -lh $PAL

    EXTRA_FLAGS=""
    ALPHA_FILE=$IMGDIR/${name}_alpha.png
    if [ -f $ALPHA_FILE ]; then
        echo "... using $ALPHA_FILE"
        EXTRA_FLAGS="--fake_alpha=$ALPHA_FILE"
    fi

    RES_FILE=$OUTDIR/${name}.RES.binary
    if [ -f $RES_FILE ]; then
        echo "RES_FILE $RES_FILE already exists"
        continue
    fi

    python -m learn.tf.color.compress_main $EXTRA_FLAGS \
           --input=$I \
           --palette=$PAL \
           --output_image=$OUTDIR/${name}.approx_image.png \
           --output_encoding=$OUTDIR/${name}.uv_encoding.png \
           --output_binary=$RES_FILE \
           --img_width=512
    #32

done
