#!/bin/sh

# EXAMPLES:
#  $0 sql 512
#  $0 java 128
#  $0 typescript 512

if [[ $# -ne 2 ]]; then
        echo "use: $0 lang vector_size, example: $0 java 128"
        exit 1
fi

LANG=$1
VEC_SIZE=$2

python preprocess.py -token_src ../../data/source/${LANG}_index_to_tokenized_code.pkl -token_tgt ../../data/source/${LANG}_index_to_tokenized_qt.pkl -split_indices ../../data/source/split_indices_${LANG}_cleaned.pkl -src_word2id ../../data/source/${LANG}.code.vocab.pkl -src_seq_length 120 -tgt_seq_length 20 -tgt_word2id ../../data/source/${LANG}.qt.vocab.pkl -save_data dataset/train_qt_new_cleaned/${LANG}.processed_all --DEV_src ../../data/source/codenn_${LANG}/codenn.dev.ix_to_tokenized_code.pkl --DEV_tgt ../../data/source/codenn_${LANG}/codenn.dev.ix_to_tokenized_qt.pkl --DEV_indices ../../data/source/codenn_${LANG}/codenn.dev.qid_cid_pair.gen.dataset.pkl --EVAL_src ../../data/source/codenn_${LANG}/codenn.eval.ix_to_tokenized_code.pkl --EVAL_tgt ../../data/source/codenn_${LANG}/codenn.eval.ix_to_tokenized_qt.pkl --EVAL_indices ../../data/source/codenn_${LANG}/codenn.eval.qid_cid_pair.gen.dataset.pkl -word_vec_size ${VEC_SIZE} > log_qt_new_cleaned/log.${LANG}.preprocess
