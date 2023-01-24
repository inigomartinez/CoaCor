"""
@author: Iñigo Martínez <inigo.martinez@tecnalia.com>
@copyright: 2022-2023 Tecnalia
"""

from datetime import datetime
from iteration_utilities import grouper
import logging
from models import *
import numpy as np
from operator import itemgetter
import os
import pickle
import random
import sys
import torch

LOGGER = logging.getLogger('PROXYA')

logging.basicConfig(
    level=logging.DEBUG,
    # format='[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s',
    format='[%(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler(f'DATIVE-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)

USE_CUDA = True
BATCH_SIZE = 50
TOKEN_LIST_SIZE = 120
SEED = 42

LANG = 'typescript'

WORK_DIR = '../data/'
MODEL_DIR = '../checkpoint/QC_valcodenn/'

VOCAB_CONF = {
    'code': f'{LANG}.code.vocab.pkl',
    'qt': f'{LANG}.qt.vocab.pkl',
}

DATA_CONF = {
    'code': f'codenn_combine_new.{LANG}.dev.code.pkl',
    'qt': f'codenn_combine_new.{LANG}.dev.qt.pkl',
}

# FIXME1: qtnwords and codenwords have to be set with qt and code vocabulary sizes
# FIXME2: optimizer_adam_lr, browdropout and seqencdropout are actually only their decimal parts, os they are 0.{value}
MODEL_CONF = {
    'java': {
        'qtlen': 20,
        'codelen': 120,
        'qtnwords': None,
        'codenwords': None,
        'batch': 64,
        'optimizer_adam_lr': '001',
        'embsize': 128,
        'lstmdims': 200,
        'bowdropout': '35',
        'seqencdropout': '35',
        'codeenc': 'bilstm',
    },
    'typescript': {
        'qtlen': 20,
        'codelen': 120,
        'qtnwords': None,
        'codenwords': None,
        'batch': 256,
        'optimizer_adam_lr': '001',
        'embsize': 768,
        'lstmdims': 400,
        'bowdropout': '35',
        'seqencdropout': '35',
        'codeenc': 'bilstm',
    },
}

# FIXME1: Used by Joint Embedder and CodennDataset
# FIXME2: qt_n_words and code_n_words have to be set with qt and code vocabulary sizes
CONF = {
    'margin': 0.05,

    'qt_n_words': None,
    'code_n_words': None,

    'emb_size': MODEL_CONF[LANG]['embsize'],
    'lstm_dims': MODEL_CONF[LANG]['lstmdims'],

    'code_encoder': MODEL_CONF[LANG]['codeenc'],

    'use_anno': False,

    'workdir': WORK_DIR,

    'qt_len': MODEL_CONF[LANG]['qtlen'],
    'code_len': MODEL_CONF[LANG]['codelen'],
    'seqenc_dropout': float(f'0.{MODEL_CONF[LANG]["seqencdropout"]}'),

    '<pad>': 0,
    'vocab_qt': VOCAB_CONF['qt'],
    'vocab_code': VOCAB_CONF['code'],

    'val_qt': f'codenn_combine_new.{LANG}.dev.qt.pkl',
    'val_code': f'codenn_combine_new.{LANG}.dev.code.pkl',
}

def _batchify(data, pad_id, align_right=False, include_lengths=False):
    lengths = [x.size(0) for x in data]
    max_length = max(lengths)
    out = data[0].new(len(data), max_length).fill_(pad_id)
    for i in range(len(data)):
        data_length = data[i].size(0)
        offset = max_length - data_length if align_right else 0
        out[i].narrow(0, offset, data_length).copy_(data[i])

    if include_lengths:
        return out, lengths
    else:
        return out

def _get_best_cand(model, group, qts_repr, pad_id):
    cands = _batchify(group, pad_id)
    cands_repr = model.cand_encoding(cands)

    preds = _predict(model, qts_repr, cands_repr)

    return cands[preds.index(0)]

def _get_better_cands(model, tensors, batch_size, qts_repr, pad_id, fill_value=None):
    return [_get_best_cand(model, group, qts_repr, pad_id)
            for group in grouper(tensors, batch_size, fillvalue=fill_value)]

def _predict(model, qt_repr, cands_repr):
    sims = model.scoring(qt_repr, cands_repr).data.cpu().numpy()
    negsims = np.negative(sims)
    predict = np.argsort(negsims)

    return list(map(int, predict))

def _predict_code(qt_tensor, code_tensors, batch_size, pad_id):
    qts = _batchify([qt_tensor] * batch_size, pad_id)
    qts_repr = model.qt_encoding(qts)

    empty_tensor = torch.LongTensor([pad_id] * TOKEN_LIST_SIZE)
    iteration = 0
    while len(code_tensors) != 1:
        logging.debug(f'iteration: {iteration} check_tensor_code: {len(code_tensors)}')
        code_tensors = _get_better_cands(model, code_tensors, BATCH_SIZE, qts_repr, pad_id, fill_value=empty_tensor)
        iteration += 1

    return code_tensors[0]

def _get_tensors(indices):
    if USE_CUDA:
        return list(map(lambda e: torch.LongTensor(e).cuda(), indices))
    return list(map(torch.LongTensor, code_indices))

def _decode(indices, vocab, end_id=None):
    if end_id is not None:
        indices = indices[:indices.index(end_id)]
    return list(map(vocab.get, indices))

def _set_eos_and_pad(tokens, pad_id, eos_id=None):
    tokens_size = len(tokens)
    if eos_id:
        if tokens_size == TOKEN_LIST_SIZE:
            tokens[-1] = eos_id
        else:
            tokens += [eos_id]
            tokens_size += 1

    if tokens_size < TOKEN_LIST_SIZE:
        tokens += ([pad_id] * (TOKEN_LIST_SIZE - tokens_size))

    return tokens

def _invert(d):
    return {v: k for (k, v) in d.items()}

def _set_seed(v):
    torch.manual_seed(v)
    np.random.seed(v)
    random.seed(v)

    if USE_CUDA:
        # cuda.set_device(opt.gpus[0])
        torch.cuda.manual_seed(v)


def _load_model(model_conf):
    model_name = '_'.join([f'{k}_{v}' for k, v in model_conf.items()])

    model = JointEmbeder(CONF)
    model_name = os.path.join(MODEL_DIR, model_name, 'best_model.ckpt')
    logging.debug(f'model: {model_name}')
    model.load_state_dict(torch.load(model_name))

    if USE_CUDA:
        model.cuda()

    return model

def _load_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def _load_vocabs(vocab_conf):
    code_vocab, qt_vocab = [_load_file(os.path.join(WORK_DIR, filename))
                            for _, filename in vocab_conf.items()]

    return code_vocab, _invert(code_vocab), qt_vocab, _invert(qt_vocab)

def _load_tokens(data_conf, pad_id, eos_id):
    code_tokens, qt_tokens = [_load_file(os.path.join(WORK_DIR, filename))
                              for _, filename in data_conf.items()]

    # code_tokens = list(set(code_tokens))
    code_indices = list(map(lambda e: _set_eos_and_pad(list(map(int, e.split())), pad_id), code_tokens))
    code_tensors = _get_tensors(code_indices)

    # qt_tokens = list(sorted(set(map(itemgetter(0), qt_tokens[::BATCH_SIZE]))))
    qt_tokens = list(map(itemgetter(0), qt_tokens[::BATCH_SIZE]))
    qt_indices = list(map(lambda e: _set_eos_and_pad(list(map(int, e.split())), pad_id, eos_id), qt_tokens))
    qt_tensors = _get_tensors(qt_indices)

    return code_tokens, code_indices, code_tensors, qt_tokens, qt_indices, qt_tensors

if __name__ == '__main__':
    _set_seed(SEED)

    code_vocab, code_vocab_indices, qt_vocab, qt_vocab_indices = _load_vocabs(VOCAB_CONF)

    CONF['qt_n_words'] = MODEL_CONF[LANG]['qtnwords'] = len(qt_vocab)
    CONF['code_n_words'] = MODEL_CONF[LANG]['codenwords'] = len(code_vocab)

    pad_id = qt_vocab.get('<pad>')
    eos_id = qt_vocab.get('<eos>')
    code_tokens, code_indices, code_tensors, qt_tokens, qt_indices, qt_tensors = _load_tokens(DATA_CONF, pad_id, eos_id)

    model = _load_model(MODEL_CONF[LANG])

    # code_tensors = code_tensors[:11000]
    qt_tensor = qt_tensors[random.randint(0, len(qt_tensors) - 1)]
    # qt_tensor = qt_tensors[0]
    code_tensor = _predict_code(qt_tensor, code_tensors, BATCH_SIZE, pad_id)
    qt = _decode(qt_tensor.tolist(), qt_vocab_indices, eos_id)
    code = _decode(code_tensor.tolist(), code_vocab_indices, pad_id)
    print(f'QT: {qt}\nCODE: {code}')
