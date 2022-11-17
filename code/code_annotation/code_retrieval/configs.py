def get_config(args):
    conf = {
        # Change it to necessary directory
        'workdir': 'dataset/cr_data/',

        'PAD': 0,
        'BOS': 1,
        'EOS': 2,
        'UNK': 3,

        'train_qt': f'{args.lang}.train.qt.pkl',
        'train_code': f'{args.lang}.train.code.pkl',

        # parameters
        'qt_len': 20,
        'code_len': 120,

        'qt_n_words': args.qt_n_words,  # 4 is added for UNK, EOS, SOS, PAD
        'code_n_words': args.code_n_words,

        # vocabulary info
        'vocab_qt': f'{args.lang}.qt.vocab.pkl',
        'vocab_code': f'{args.lang}.code.vocab.pkl',

        # model
        'checkpoint': f'dataset/cr_data/qtlen_20_codelen_120_qtnwords_{args.qt_n_words}_codenwords_{args.code_n_words}'
                      '_batch_256_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35'
                      '_codeenc_bilstm/best_model.ckpt',
        'use_anno': 0,
        'emb_size': 200,
        'lstm_dims': 400,
        'margin': 0.05,
        'code_encoder': 'bilstm',
        'bow_dropout': 1.0,
        'seqenc_dropout': 1.0
    }
    return conf
