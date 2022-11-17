#!/bin/sh

# EXAMPLES:
#  $0 sql
#  $0 java
#  $0 typescript

if [[ $# -ne 1 ]]; then
        echo "use: $0 lang, example: $0 java"
        exit 1
fi

LANG=$1

python run.py test_a2c ${LANG} 0 20 1 bleu dataset/result_${LANG}_qt_new_cleaned/model_rf_hasBaseline_attn1_brnn1_decay45_dropout0.5_Sentcr_reinforce/model_rf_hasBaseline_attn1_brnn1_decay45_dropout0.5_Sentcr_reinforce_57.pt codenn_all 1 1 1
