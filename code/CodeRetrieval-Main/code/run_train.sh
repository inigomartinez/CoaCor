#!/bin/sh

# EXAMPLES:
#  $0 sql 200 400 256 7775 7726
#  $0 java 128 400 64 176946 1907029
#  $0 typescript 512 400 256 1180 2643

if [[ $# -ne 6 ]]; then
        echo "use: $0 lang vector_size lstm_dims batch_size qt_n_words code_n_words, example: $0 java 128 400 256 176946 1907029"
        exit 1
fi

LANG=$1
VEC_SIZE=$2
LSTM_DIMS=$3
BATCH_SIZE=$4
QT_N_WORDS=$5
CODE_N_WORDS=$6

# train QC
python codesearcher.py --lang ${LANG} --mode train --qt_n_words ${QT_N_WORDS} --code_n_words ${CODE_N_WORDS} --anno_n_words ${QT_N_WORDS} --dropout 0.35 --emb_size ${VEC_SIZE} --lstm_dims ${LSTM_DIMS} --batch_size ${BATCH_SIZE} --val_setup codenn --use_anno 0 --reload 0 > log/qc_${LANG}_train_drop35_emb${VEC_SIZE}_lstm${LSTM_DIMS}_bs${BATCH_SIZE}.log

# train QN-RL-MRR
python codesearcher.py --lang ${LANG} --mode train --qt_n_words ${QT_N_WORDS} --code_n_words ${CODE_N_WORDS} --anno_n_words ${QT_N_WORDS} --dropout 0.35 --emb_size ${VEC_SIZE} --lstm_dims ${LSTM_DIMS} --batch_size ${BATCH_SIZE} --val_setup codenn --use_anno 0 --reload 0 --qn_mode rl_mrr > log/qn_${LANG}_rlmrr_train_drop35_emb${VEC_SIZE}_lstm${LSTM_DIMS}_bs${BATCH_SIZE}.log
