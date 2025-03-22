#!/bin/bash -e

set -o nounset

if [ -z ${COLOR_EXP_BASE+x} ]; then
    echo "Must export $COLOR_EXP_BASE variable"
    exit 1
fi

if [ -z ${COLOR_EVAL_FLAGS+x} ]; then
    echo "No custom COLOR_EVAL_FLAGS specified in the env"
    COLOR_EVAL_FLAGS=""
else
    echo "Using env COLOR_EVAL_FLAGS: ${COLOR_EVAL_FLAGS}"
fi

EXP_GROUP=$1
MODEL=$2
BASE_OUT_DIR=$3  # will create exp_group/exp_name here
INPUT=$4  # e.g. source alpha_datasets.sh
SUFFIX=$5
DATA_DIR=/ais/gobi5/shumash/Data/Color/images512

MODEL_DIR=${COLOR_EXP_BASE}/${EXP_GROUP}/${MODEL}
OUT_DIR=${BASE_OUT_DIR}/${EXP_GROUP}/${MODEL}/results_${SUFFIX}
LOG=$OUT_DIR/results_log.txt

echo "Using flags from $MODEL_DIR"
if [ ! -f $MODEL_DIR/scripts/export_flags.sh ]; then
    echo "Error: must create $MODEL_DIR/scripts/export_flags.sh"
    echo "with export RUN_FLAGS=... linke in run.sh"
    exit 1
fi

source $MODEL_DIR/scripts/export_flags.sh
export RUN_FLAGS="${RUN_FLAGS} ${COLOR_EVAL_FLAGS}"

echo "Creating out dir: $OUT_DIR"
mkdir -p $OUT_DIR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR
srun --gres=gpu:1 -c 2 -p gpuc \
     --output=$LOG \
     base_run.sh \
     -r ${OUT_DIR} \
     -t ${INPUT} \
     -m $MODEL_DIR/out \
     -d $DATA_DIR \
     -u

echo "Output here: $LOG"
