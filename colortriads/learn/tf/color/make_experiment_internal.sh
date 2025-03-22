#!/bin/bash -e

set -o nounset

# NEVER RUN THIS STANDALONE
# This script should be executed from inside make_experiment.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z ${DATA_DIR+x} ]; then
    echo "Setting data automatically"

    if [ "$(hostname)" == "cluster58" ]; then
        echo "Setting up on cluster58"
        BASE_DATA_DIR=/ais/gobi5/shumash/Data/Color/
        DATA_DIR=${BASE_DATA_DIR}/images512
        SPLIT_DIR=${BASE_DATA_DIR}/splits/splits512

        # TODO(shumash): add an option to weigh GAP data by colorfulness
        TRAIN=${SPLIT_DIR}/img_splits/train/target.txt
        if [ "${GAP_REWEIGHTED}" -eq 0 ]; then
            echo "--> Using GAP unreweighted data for 50% of input"
            TRAIN=${TRAIN},${SPLIT_DIR}/img_splits/train/GAP.txt
        else
            echo "--> Using GAP **reweighted** data for 50% of input"
            TRAIN=${TRAIN},${SPLIT_DIR}/img_splits/train/GAP_reweighted.txt
        fi

        if [ "${INPUT_MODE}" -eq 4 ]; then
            E=${SPLIT_DIR}/patch_splits/val/target_easy_patches.txt
            M=${SPLIT_DIR}/patch_splits/val/target_medium_patches.txt
            H=${SPLIT_DIR}/patch_splits/val/target_hard_patches.txt
            P=${DATA_DIR}/aux/pretty_patches_unscaled/pretty_patches.txt
            EVAL=${E},${M},${H},${P}

            E=${SPLIT_DIR}/patch_splits/test/target_easy_patches.txt
            M=${SPLIT_DIR}/patch_splits/test/target_medium_patches.txt
            H=${SPLIT_DIR}/patch_splits/test/target_hard_patches.txt
            TEST=${E},${M},${H},${P}
        else
            FA=${SPLIT_DIR}/img_splits/val/fine_art.txt
            GD=${SPLIT_DIR}/img_splits/val/graphic_design.txt
            PH=${SPLIT_DIR}/img_splits/val/photo.txt
            VIZ=${SPLIT_DIR}/img_splits/val/viz.txt
            GAP=${SPLIT_DIR}/img_splits/val/GAP_sm.txt
            EVAL=${FA},${GD},${PH},${VIZ},${GAP}

            FA=${SPLIT_DIR}/img_splits/test/fine_art.txt
            GD=${SPLIT_DIR}/img_splits/test/graphic_design.txt
            PH=${SPLIT_DIR}/img_splits/test/photo.txt
            VIZ=${SPLIT_DIR}/img_splits/test/viz.txt
            GAP=${SPLIT_DIR}/img_splits/test/GAP_sm.txt
            TEST=${FA},${GD},${PH},${VIZ},${GAP}
        fi
    else
        echo "Setting up locally or elsewhere (just for a quick test of the setup)"

        DATASET=split0 #all_imgs #split0 #gap_mediums_oil_split0

        DATA_DIR=${HOME}/Data/Images
        DATA_LIST_DIR=${SCRIPT_DIR}/../../../../experiments/color/data
        TRAIN=${DATA_LIST_DIR}/${DATASET}/keys_train.txt
        EVAL=${DATA_LIST_DIR}/${DATASET}/keys_eval.txt
        TEST=${DATA_LIST_DIR}/${DATASET}/keys_test.txt

        if [ "${INPUT_MODE}" -eq 4 ]; then   # special input for patches
            echo "Using special test patches"
            EVAL=${DATA_LIST_DIR}/patch_splits/val/easy.txt,${DATA_LIST_DIR}/patch_splits/val/medium.txt,${DATA_LIST_DIR}/patch_splits/val/hard.txt,${DATA_LIST_DIR}/test_patches/pretty.txt
            TEST=${DATA_LIST_DIR}/patch_splits/val/easy.txt,${DATA_LIST_DIR}/patch_splits/val/medium.txt,${DATA_LIST_DIR}/patch_splits/val/hard.txt,${DATA_LIST_DIR}/test_patches/pretty.txt
            TRAIN=${DATA_LIST_DIR}/patch_splits/train.txt
        fi
    fi
fi


RUN_DIR_PREFIX=${RUN_LOC}/${EXP_GROUP}/${EXP_PREFIX}

if [ "${PALETTE_ENCODER_MODE}" -eq 4 ]; then
    PALETTE_LAYER_SPECS=${PG_CONV_OPTS}
else
    PALETTE_LAYER_SPECS=${PG_FC_SIZES}
fi

# Avoid nounset failure when EXTRA_FLAGS not set
if [ -z ${EXTRA_FLAGS+x} ]; then
    EXTRA_FLAGS=""
fi

EXTRA_FLAGS="${EXTRA_FLAGS} \\
--max_tri_subdivs=${MAX_SUBDIVS} \\
--encoder_mode=${ENCODER_MODE} \\
--palette_graph_mode=${PALETTE_ENCODER_MODE} \\
--wind_num_channels=${WIND_NCHANNELS}"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--learning_rate=${LEARNING_RATE}"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--img_width=${IMG_WIDTH}"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--max_colors=${MAX_COLORS}"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--palette_layer_specs='${PALETTE_LAYER_SPECS}'"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--conv_filter_sizes=${CONV_FSIZES} \\
--conv_filter_counts=${CONV_FCOUNTS}"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--restore_palette_graph=${PRELOAD_PALETTE_GRAPH}"

#AMLAN ADDITIONS
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--mask_softmax_temp='${MASK_SOFTMAX_TEMP}'"
EXTRA_FLAGS="${EXTRA_FLAGS} \\
--mask_dropout='${MASK_DROPOUT}'"

CACHE_DIR="" #${SCRIPT_DIR}/../../../../experiments/color/datacaches

if [ "${GAP_REWEIGHTED}" -eq 1 ]; then
    RUN_DIR_PREFIX=${RUN_DIR_PREFIX}_rw
fi
RUN_DIR=${RUN_DIR_PREFIX}_gr${ENCODER_MODE}w${WIND_NCHANNELS}
if [ "${INPUT_MODE}" -eq 0 ]; then
    echo "Training on Histograms ~~~~~~~~~~~~~~~~~~"
    FIELD=1
    DATA_DIR=${SCRIPT_DIR}/../../../..
    RUN_DIR=${RUN_DIR}_hist
elif [ "${INPUT_MODE}" -eq 1 ]; then
    echo "Training on Histograms Computed on the Fly ~~~~~~~~~~~~~~~~~~"
    FIELD=0
    RUN_DIR=${RUN_DIR}_histcomp
elif [ "${INPUT_MODE}" -eq 2 -o "${INPUT_MODE}" -eq 4 ]; then
    echo "Training on Images ~~~~~~~~~~~~~~~~~~"
    FIELD=0
    RUN_DIR=${RUN_DIR}_img${IMG_WIDTH}
    if [ "${BW_INPUT}" != "0" ]; then
        echo "Using black and white input"
        RUN_DIR=${RUN_DIR}bw${GLIMPSES}hints
        EXTRA_FLAGS="${EXTRA_FLAGS} \\
--frac_bw_input=${BW_INPUT}"
    else
        RUN_DIR=${RUN_DIR}rgb
    fi
else
    echo "Training on Synthetic Palette Images ~~~~~~~~~~~~~~~~~~"
    FIELD=0
    DATA_LIST_DIR=${SCRIPT_DIR}/../../../../experiments/color/data/synthpal0
    RUN_DIR=${RUN_DIR}_synthpalimg
fi

if [ "${PATCH_WIDTH}" -gt 0 ]; then
    RUN_DIR=${RUN_DIR}_pwidth${PATCH_WIDTH}
fi
RUN_DIR=${RUN_DIR}_colors${MAX_COLORS}_subdiv${MAX_SUBDIVS}

if [ "${ENABLE_LEVELS}" -gt 0 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} \\
--encode_subdiv_levels=True"
    RUN_DIR=${RUN_DIR}_levels
fi

if [ "${ENABLE_ALPHA}" -gt 0 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} \\
--encode_alpha=True"
    RUN_DIR=${RUN_DIR}_alpha
fi


echo "Run directory: ${RUN_DIR}"

if [ -d ${RUN_DIR} ]; then
    echo "Really override run dir: ${RUN_DIR}? [y/n]"
    read YESORNO
    if [ "${YESORNO}" == "y" ]; then
        rm -rf ${RUN_DIR}/*
    else
        echo "Aborting..."
        exit 1
    fi
fi

EXTRA_FLAGS="${EXTRA_FLAGS} \\
--data_input_field=${FIELD} \\
--input_mode=${INPUT_MODE} \\
--patch_width=${PATCH_WIDTH}"


mkdir -p ${RUN_DIR}/scripts
mkdir -p ${RUN_DIR}/out

cat > ${RUN_DIR}/scripts/train.sh<<EOF
#!/bin/bash -e

set -o nounset

cd ${SCRIPT_DIR}/../../..
python -m learn.tf.color.train_main ${EXTRA_FLAGS} \\
--max_test_batch_size=${MAX_TESTBATCH_SIZE} \\
--losses=${LOSSES} \\
--loss_weights=${LOSS_WEIGHTS} \\
--data_dir=${DATA_DIR} \\
--cache_dir=${CACHE_DIR} \\
--traindata=${TRAIN} \\
--testdata=${EVAL} \\
--run_dir=${RUN_DIR}/out \\
--its=100000
#--debug=True
EOF
chmod a+x ${RUN_DIR}/scripts/train.sh
NAMED_SCRIPT=${RUN_DIR}/scripts/$(basename ${RUN_DIR})_train.sh
ln -s ${RUN_DIR}/scripts/train.sh ${NAMED_SCRIPT}


cat > ${RUN_DIR}/scripts/train_cluster.sh<<EOF
#!/bin/bash -e

set -o nounset

OVERRIDE=0
USAGE="Run script on the cluster.
Simply call script with -o to overwrite current dir."

while getopts ':ho' option; do
    case "\$option" in
        h) echo "\$USAGE"
           exit
           ;;
        o) OVERRIDE=1
           echo "Overriding existing output"
           ;;
        \?) printf "ERROR! illegal option: -%s\n" "\$OPTARG" >&2
            echo "\$USAGE" >&2
            exit 1
            ;;
    esac
done
shift \$((OPTIND - 1))

if [ "\$OVERRIDE" -eq 1 ]; then
   echo 'Really overwrite training run? [y/n]'
   read YN
   if [ "\$YN" == "y" ]; then
     rm -rf ${RUN_DIR}/out/*
   fi
fi

LOG=${RUN_DIR}/out/srun_log_\${RANDOM}.txt
LATEST_LOG=${RUN_DIR}/out/latest_srun_log.txt
touch \$LOG
touch \${LATEST_LOG}
rm \${LATEST_LOG}
ln -s \$LOG \${LATEST_LOG}
srun --gres=gpu:1 -c 2 -p gpuc \\
--output=\$LOG \\
${NAMED_SCRIPT} &

echo "Running: ${NAMED_SCRIPT}"
echo "Output: ${RUN_DIR}/out/latest_srun_log.txt"

EOF
chmod a+x ${RUN_DIR}/scripts/train_cluster.sh


RUN_LOSSES=$LOSSES
RUN_LOSS_WEIGHTS=$LOSS_WEIGHTS

if [ "$INPUT_MODE" -eq 4 ]; then
    echo "Adding Reconstruction percent to losses for evaluation"
    RUN_LOSSES="${RUN_LOSSES},RECON_PERCENT"
    RUN_LOSS_WEIGHTS="${RUN_LOSS_WEIGHTS},0.0"
fi

RUN_FLAGS="${EXTRA_FLAGS} \\
--losses=${RUN_LOSSES} \\
--loss_weights=${RUN_LOSS_WEIGHTS}"

ODIR=/tmp/out/$(basename ${RUN_DIR})
cat > ${RUN_DIR}/scripts/run.sh<<EOF
#!/bin/bash -e

set -o nounset

USAGE="Run evaluation procedure, in one of two modes: 'test', 'eval'.
Finds the model relative to this script's location; DO NOT MOVE IT."

if [ \$# -le 0 ]; then
    echo "Error: no mode specified"
    printf "\$USAGE"
    exit
fi

MODE=\$1
if [ "\$MODE" == "eval" ]; then
   echo "Running in eval mode"
   TESTSETS=${EVAL}
elif [ "\$MODE" == "test" ]; then
   echo "Running in test mode"
   TESTSETS=${TEST}
else
    echo "Unknown mode: \$MODE"
    exit 1
fi

SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
RUN_DIR=\${SCRIPT_DIR}/../out
OUT_DIR=\${SCRIPT_DIR}/../res_\$MODE

export RUN_FLAGS="${RUN_FLAGS}"

if [ "\$(basename \$(pwd))" == "python" ]; then
CODE_LOC=learn/tf/color
else
CODE_LOC=${SCRIPT_DIR}
fi

\${CODE_LOC}/base_run.sh \
-r \${OUT_DIR} \
-t \${TESTSETS} \
-m \${RUN_DIR} \
-d ${DATA_DIR} \
-u

EOF
chmod a+x ${RUN_DIR}/scripts/run.sh

NAMED_SCRIPT=${RUN_DIR}/scripts/R$(basename ${RUN_DIR})_run.sh
ln -s ${RUN_DIR}/scripts/run.sh ${NAMED_SCRIPT}
cat > ${RUN_DIR}/scripts/cluster_run.sh<<EOF
#!/bin/bash -e

set -o nounset

if [ \$# -le 0 ]; then
    echo "Error: no mode specified"
    exit
fi

MODE=\$1

SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
RUN_DIR=\${SCRIPT_DIR}/../out
OUT_DIR=\${SCRIPT_DIR}/../res_\$MODE
LOG=\${OUT_DIR}/results_log.txt

mkdir -p \$OUT_DIR

srun --gres=gpu:1 -c 2 -p gpuc \\
--output=\$LOG \\
${NAMED_SCRIPT} \$MODE &

echo "Running: ${NAMED_SCRIPT}"
echo "Results: \${OUT_DIR}"
echo "Log: \$LOG"

EOF
chmod a+x ${RUN_DIR}/scripts/cluster_run.sh

echo "Wrote train and run scripts to ${RUN_DIR}/scripts"

echo "To train model:"
echo ${RUN_DIR}/scripts/train_cluster.sh


# Data file generation notes
#cat experiments/color_ana/counts1/digital_painting_rgb10/key_file.txt | awk '{if ($2 ~ /0.txt/) {print;}}' > experiments/color_ana/counts1/digital_painting_rgb10/keys_test.tx
#cat experiments/color_ana/counts1/digital_painting_rgb10/keys_test.txt | awk '{gsub(/.*\/digital_painting\/*/, ""); print $1 " " $1;}' > experiments/color_ana/img_data_split0/keys_test.txt
