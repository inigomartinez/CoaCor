def get_config(args):
    conf = {
        # data_params
        'workdir': '../data/',
        'ckptdir': '../checkpoint/',

        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3,

        'train_qt': f'{args.lang}.train.qt.pkl',
        'train_code': f'{args.lang}.train.code.pkl',

        'val_qt': f'{args.lang}.val.qt.pkl',
        'val_code': f'{args.lang}.val.code.pkl',

        'test_qt': f'{args.lang}.test.qt.pkl',
        'test_code': f'{args.lang}.test.code.pkl',

        'qt_len': 20,
        'code_len': 120,
        'anno_len': 20,

        'qt_n_words': args.qt_n_words,  # 4 is added for UNK, EOS, SOS, PAD
        'code_n_words': args.code_n_words,
        'anno_n_words': args.anno_n_words,

        # vocabulary info
        'vocab_qt': f'{args.lang}.qt.vocab.pkl',
        'vocab_code': f'{args.lang}.code.vocab.pkl',
        'vocab_anno': f'{args.lang}.qt.vocab.pkl',

        # training_params
        'use_anno': args.use_anno,

        'batch_size': args.batch_size,
        'nb_epoch': 100,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 50,
        'save_every': 10,
        'patience': 20,
        'reload': 0,  # reload>0, model is reloaded.

        # model_params
        'emb_size': args.emb_size,
        # recurrent
        'lstm_dims': args.lstm_dims,
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'margin': 0.05,
        'code_encoder': 'bilstm',  # bow, bilstm
    }

    if conf['use_anno']:
        if args.qn_mode == "codenn_gen":
            conf['vocab_anno'] = f'{args.lang}.ga.vocab.pkl'
            conf['anno_n_words'] = 827

            conf['train_anno'] = f'{args.lang}.train.ga.pkl'

            if (args.mode == "train" and args.val_setup == "codenn") or (
                args.mode in {"eval", "collect"} and args.eval_setup == "codenn"):
                conf['val_qt'] = f'codenn_combine_new.{args.lang}.dev.qt.pkl'
                conf['val_anno'] = f'codenn.{args.lang}.dev.ga.pkl'
                conf['test_qt'] = f'codenn_combine_new.{args.lang}.eval.qt.pkl'
                conf['test_anno'] = f'codenn.{args.lang}.eval.ga.pkl'
            else:
                conf['val_anno'] = f'{args.lang}.valid.ga.pkl'
                conf['test_anno'] = f'{args.lang}.test.ga.pkl'

        else:
            conf['train_anno'] = f'{args.lang}.train.anno_%s.pkl' % args.qn_mode
            if (args.mode == "train" and args.val_setup == "codenn") or (
                args.mode in {"eval", "collect"} and args.eval_setup == "codenn"):
                conf['val_qt'] = f'codenn_combine_new.{args.lang}.dev.qt.pkl'
                conf['val_anno'] = f'codenn_combine_new.{args.lang}.dev.anno_%s.pkl' % args.qn_mode
                conf['test_qt'] = f'codenn_combine_new.{args.lang}.eval.qt.pkl'
                conf['test_anno'] = f'codenn_combine_new.{args.lang}.eval.anno_%s.pkl' % args.qn_mode
            else:
                conf['val_anno'] = f'{args.lang}.val.anno_%s.pkl' % args.qn_mode
                conf['test_anno'] = f'{args.lang}.test.anno_%s.pkl' % args.qn_mode
    else:
        if (args.mode == "train" and args.val_setup == "codenn") or (
            args.mode in {"eval", "collect"} and args.eval_setup == "codenn"):
            conf['val_qt'] = f'codenn_combine_new.{args.lang}.dev.qt.pkl'
            conf['val_code'] = f'codenn_combine_new.{args.lang}.dev.code.pkl'
            conf['test_qt'] = f'codenn_combine_new.{args.lang}.eval.qt.pkl'
            conf['test_code'] = f'codenn_combine_new.{args.lang}.eval.code.pkl'

    return conf
