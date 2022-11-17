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

python run.py train_a2c ${LANG} 0 1 20 0 cr dataset/result_${LANG}_qt_new_cleaned/model_xent_attn1_brnn1_decay15_emb${VEC_SIZE}_rnn${LSTM_DIMS}_dropout0.5/model_xent_attn1_brnn1_decay15_emb${VEC_SIZE}_rnn${LSTM_DIMS}_dropout0.5_20.pt 15 64 10 1 1 45 0.0001 ${VEC_SIZE} ${LSTM_DIMS} 0.5 ${BATCH_SIZE} 0 1
