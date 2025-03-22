#!/bin/bash -e

set -o nounset

USAGE="Run evaluation procedure, in one of two modes: 'test', 'eval'.
Expects all run flags to be set in an external var RUN_FLAGS,
except for the ones below (must be specified).

Invocation:
run.sh <flags>

Flags:
-h show help
-t set test set
-r set result directory
-m sets model directory
-d set data dir"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SAFE=1
while getopts ':ht:r:d:m:u' option; do
    case "$option" in
        h) printf "$USAGE"
           exit
           ;;
        t) TESTSETS=$OPTARG
           echo "Set testset: $TESTSETS"
           ;;
        r) ODIR=$OPTARG
           echo "Set result dir: $ODIR"
           ;;
        d) DATA_DIR=$OPTARG
           echo "Set data dir: $DATA_DIR"
           ;;
        m) MODEL_DIR=$OPTARG
           echo "Set model dir: $MODEL_DIR"
           ;;
        u) SAFE=0
           echo "No asking for confirmation"
           ;;
        \?) printf "ERROR! illegal option: -%s\n" "$OPTARG" >&2
            echo "$USAGE" >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data dir does not exist, set manually with -d?"
    echo $DATA_DIR
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Run dir does not exist; was this script moved?"
    echo $MODEL_DIR
fi

echo "RUN SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Model: $MODEL_DIR"
echo "Testsets: $TESTSETS"
echo "Outputdir: $ODIR"
echo ""
if [ "${SAFE}" -eq "1" ]; then
    echo "Continue [y/n]?"
    read YN
    if [ "$YN" != "y" -a "$YN" != "Y" ]; then
        echo "Aborting ..."
        exit
    fi
fi

mkdir -p $ODIR

echo "Running ... "

cd ${SCRIPT_DIR}/../../..
python -m learn.tf.color.run_main ${RUN_FLAGS} \
       --data_dir=${DATA_DIR} \
       --testdata=${TESTSETS} \
       --run_dir=${MODEL_DIR} \
       --output_dir=${ODIR}

echo "Done"
echo "Results written to: $ODIR"
