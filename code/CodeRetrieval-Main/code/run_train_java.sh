#!/bin/sh

# train QC
# python codesearcher.py --lang java --mode train --qt_n_words 176946 --code_n_words 1907029 --anno_n_words 176946 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 0 --reload 0 | tee log/qc_java_train_drop35_emb200_lstm400_bs256.log
python codesearcher.py --lang java --mode train --qt_n_words 176946 --code_n_words 1907029 --anno_n_words 176946 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 0 --reload 0 > log/qc_java_train_drop35_emb200_lstm400_bs256.log

# train QN-RL-MRR
# python codesearcher.py --lang java --mode train --qt_n_words 176946 --code_n_words 1907029 --anno_n_words 176946 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 1 --reload 0 --qn_mode rl_mrr | tee log/qn_java_rlmrr_train_drop35_emb200_lstm400_bs256.log
# python codesearcher.py --lang java --mode train --qt_n_words 176946 --code_n_words 1907029 --anno_n_words 176946 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 1 --reload 0 --qn_mode rl_mrr > log/qn_java_rlmrr_train_drop35_emb200_lstm400_bs256.log
python codesearcher.py --lang java --mode train --qt_n_words 176946 --code_n_words 1907029 --anno_n_words 176946 --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 0 --reload 0 --qn_mode rl_mrr > log/qn_java_rlmrr_train_drop35_emb200_lstm400_bs256.log
