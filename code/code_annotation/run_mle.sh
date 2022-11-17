#!/bin/sh

# EXAMPLES:
#  $0 sql 512 400 64
#  $0 java 128 400 64
#  $0 typescript 512 400 64

if [[ $# -ne 4 ]]; then
        echo "use: $0 lang vector_size lstm_dims batch_size, example: $0 java 128 400 64"
        exit 1
fi

LANG=$1
VEC_SIZE=$2
LSTM_DIMS=$3
BATCH_SIZE=$4

python run.py train_a2c ${LANG} 0 0 20 0 bleu None None 20 0 1 1 15 0.001 ${VEC_SIZE} ${LSTM_DIMS} 0.5 ${BATCH_SIZE} 0 1
