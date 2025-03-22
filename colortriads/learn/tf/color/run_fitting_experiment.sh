#!/bin/bash -e

set -o nounset

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR/../../..

#TEST_IMAGES=/Users/shumash/Documents/Coding/Animation/animation/data/color/test_images/basic_onesail
#BASE_ODIR=/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/basic_results0

TEST_IMAGES=$1
BASE_ODIR=$2

DEEP_PALETTES=$BASE_ODIR/palettes_deep

if [ ! -d $DEEP_PALETTES ]; then
    echo "Fitting palettes using a neural network"
    mkdir -p $BASE_ODIR/palettes_deep
    ./learn/tf/color/fit_palettes_all.sh $TEST_IMAGES $DEEP_PALETTES


    echo "Fitting palettes using matlab (do in matlab)"
    echo "Tuning palettes using matlab (do in matlab)"
    echo "After running matlab; rerun this script"
else

    if [ -f $BASE_ODIR/palettes_deeptune/init_losses.txt ]; then
        cp $BASE_ODIR/palettes_deeptune/init_losses.txt $BASE_ODIR/palettes_deep/losses.txt
    fi
    for X in 'palettes' 'palettes_deep' 'palettes_deeptune' 'palettes_16' 'palettes_deeptune16' 'palettes_32'; do

        OBNAME=$(echo $X | awk '{gsub(/palette/, "mapping"); printf "%s", $1;}')
        PALETTE_DIR=$BASE_ODIR/$X
        MAPPING_ODIR=$BASE_ODIR/$OBNAME

        if [ ! -d $PALETTE_DIR ]; then
            echo "Palettes DNE: $X"
            continue
        fi

        
        echo "Mapping palettes: $X"
        echo "$PALETTE_DIR --> $MAPPING_ODIR"
        mkdir -p $MAPPING_ODIR
        ./learn/tf/color/remap_compress_all.sh $TEST_IMAGES $PALETTE_DIR $MAPPING_ODIR

        if [ -f $PALETTE_DIR/losses.txt ]; then
            echo "LOSSES $X"
            cat $PALETTE_DIR/losses.txt | grep -v "kl_loss" | \
                awk 'BEGIN{s=0;c=0; s_kl=0; t=0}{c=c+1;s=s+$5;s_kl += $4;t=t+$6;}END{printf"Ave Eperc %0.5f, ave KL %0.5f, ave time %0.5f (count %d)\n", s/c, s_kl/c, t/c, c}'
        fi
    done
fi

# ./learn/tf/color/run_fitting_experiment.sh $TEST_IMAGES $PFITS/basic_results0
