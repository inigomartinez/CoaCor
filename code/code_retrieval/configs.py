def get_config():
    conf = {

        # Change it to necessary directory
        'workdir': './data/',

        'test_qt': 'sql.test.qt.pkl',
        'test_code': 'sql.test.code.pkl',
        'test_qb': 'sql.test.qb.pkl',

        # parameters
        'qt_len': 14,  # 9,
        'qb_len': 82,  # 44,
        'code_len': 119,  # 60,

        'qt_n_words': 4947,  # 4 is added for UNK, EOS, SOS, PAD
        'qb_n_words': 38008,
        'code_n_words': 7734,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',
        'vocab_qb': 'sql.qb.vocab.pkl',

        # training_params
        'batch_size': 24,
        'nb_epoch': 500,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 1000,
        'save_every': 10,
        'patience': 20,
        'reload': -1,  # 970,#epoch that the model is reloaded from . If reload=0, then train from scratch

        # model_params
        'emb_size': 100,
        # 'n_hidden': 400,#number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 200,  # * 2
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'init_embed_weights_qt': None,  # word2vec_100_qt.h5,
        'init_embed_weights_code': None,  # 'word2vec_100_code.h5',
        'init_embed_weights_qb': None,  # 'word2vec_100_qb.h5',
        'margin': 0.05,
        'sim_measure': 'cos',  # similarity measure: gesd, cosine, aesd
        'code_encoder': 'bilstm',  # bow,bilstm
        'use_qb': 1
    }
    return conf