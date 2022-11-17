#!/bin/sh

# python run.py train_a2c java 0 1 20 0 cr dataset/result_java_qt_new_cleaned/model_xent_attn1_brnn1_decay15_emb256_rnn256_dropout0.5/model_xent_attn1_brnn1_decay15_emb256_rnn256_dropout0.5_14.pt 15 64 10 1 1 45 0.0001 256 256 0.5 64 0 1
python a2c-train.py -lang java -data dataset/train_qt_new_cleaned/java.processed_all.train.pt -save_dir dataset/result_java_qt_new_cleaned/ -max_predict_length 20 -predict_mask 0 -end_epoch 64 -critic_pretrain_epochs 10 -sent_reward cr -has_attn 1 -has_baseline 1 -start_decay_at 45 -word_vec_size 256 -rnn_size 256 -dropout 0.5 -batch_size 64 -layers 1 -gpus 0 -brnn -start_reinforce 15 -load_from dataset/result_java_qt_new_cleaned/model_xent_attn1_brnn1_decay15_emb256_rnn256_dropout0.5/model_xent_attn1_brnn1_decay15_emb256_rnn256_dropout0.5_14.pt -show_str _attn1_brnn1_decay45_emb256_rnn256_dropout0.5_Sentcr > log_qt_new_cleaned/log.java.a2c-train_RLhasBaseline_15_64_10_attn1_brnn1_decay45_emb256_rnn256_dropout0.5_Sentcr
