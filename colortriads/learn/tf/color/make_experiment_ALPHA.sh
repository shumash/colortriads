#!/bin/bash -e

set -o nounset

# This script generates an experiment, configured by a lot of different params.
# Experiment then can be trained/run/etc.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export RUN_LOC=${SCRIPT_DIR}/../../../../experiments/color/runs

# Settings ---------------------------------------------------------------------
export GAP_REWEIGHTED=1

export INPUT_MODE=2 # Pertinent categories: 4 - patches, 2 - images
# Pertinent encoder modes: 0 - FC encode hist, 2-vanilla unet, 3-vgg unet, 4 - conv3 encode hist
# (see setup_args.py)
export ENCODER_MODE=2 # Pertinent modes: 0 - FC encode hist, 4 - conv3 encode hist (see setup_args.py)
export PALETTE_ENCODER_MODE=0  # If pre-loading palette, must set the ENCODER_MODE palette graph was trained with

export PATCH_WIDTH=-1 #32 #-1 #64 # -1
export MAX_COLORS=3  # Always 3 for now
export MAX_SUBDIVS=4  # Specified palette decoder subdivisions
export ENABLE_LEVELS=0
export ENABLE_ALPHA=0
export WIND_NCHANNELS=3  # Whether or not to use wind
export MAX_TESTBATCH_SIZE=20
export IMG_WIDTH=256 #512 # 256 - for vanilla untet, 224 for vgg unet, 512 for patches

export PG_FC_SIZES="700;200;50"
export PG_CONV_OPTS="3,64,2;3,128,2;3,256,3;1,64,1"
export CONV_FSIZES="4,4,4"
export CONV_FCOUNTS="64,128,256"
export LOSSES="L2RGB,ALPHA_TV" #,ALPHA_NEG_MAX" #,ALPHA_NEG_MAX,ALPHA_TV"  #"KL" #"L2RGB,REG,ALPHA_TV,ALPHA_BINARY"
export LOSS_WEIGHTS="1.0,0.004" #,0.001,0.0" #,0.0,0.0" #"1.0,0.0001,0.0002"
export GLIMPSES=0
export BW_INPUT=0 #0.25
export LEARNING_RATE=0.001
# my temperatures for alphas 2,3: [2.56, 3.98, 1.0] 2.84, 2.13
# *your temperatures for alphas 4,5*: 2.03, 3.54, 3.64, 3.81
export MASK_SOFTMAX_TEMP=3.81
export PRELOAD_PALETTE_GRAPH="${RUN_LOC}/palette_graphs4_reg/patchreg_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6/out"
export NALPHAS=5
#"${RUN_LOC}/palette_graphs4/pg_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6/out"
#${RUN_LOC}/tf12/palette_recon_graph0_img512_pwidth64_colors3_subdiv3
#${RUN_LOC}/tf11/test_graph0_img512_pwidth64_colors3_subdiv3
#tf9/test_graph0_img512_pwidth64_colors3_subdiv3


export EXTRA_FLAGS="--n_masks=${NALPHAS}"


export EXP_GROUP="NALPHA_run0"  # will create an experiment subdir in your RUN_LOC
export EXP_PREFIX="a${NALPHAS}t${MASK_SOFTMAX_TEMP}"
# this is the prefix of your job; can set based on params being experimented on
# End of Settings --------------------------------------------------------------

# Optionally set:
# DATA_DIR
# EVAL
# TEST
# TRAIN

${SCRIPT_DIR}/make_experiment_internal.sh
