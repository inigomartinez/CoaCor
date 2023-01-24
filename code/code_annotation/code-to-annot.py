"""
@author: Iñigo Martínez <inigo.martinez@tecnalia.com>
@copyright: 2022-2023 Tecnalia
"""

from datetime import datetime
import logging
import numpy as np
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
N_PREDS = 5

LANG = 'typescript'

WORK_DIR = '../../data/source'

DATA_FILENAME = f'dataset/train_qt_new_cleaned/{LANG}.processed_all.train.pt'

MODEL_BASE_DIR = f'dataset/result_{LANG}_qt_new_cleaned/'
MODEL_BASE_ID = 'model_rf_hasBaseline_{}_Sentcr_reinforce'

MODEL_NUMBER = {
    'java': 58,
    'typescript': 64,
}

VOCAB_CONF = {
    'code': f'{LANG}.code.vocab.pkl',
    'qt': f'{LANG}.qt.vocab.pkl',
}

DATA_CONF = {
    'code': f'{LANG}_index_to_tokenized_code.pkl',
    'qt': f'{LANG}_index_to_tokenized_qt.pkl',
    'split': f'split_indices_{LANG}_cleaned.pkl',
}

MODEL_CONF = {
    'typescript': {
        'attn': 1,
        'brnn': 1,
        'decay': 45,
        'emb': 768,
        'rnn': 400,
        'dropout': 0.5,
    },
    'java': {
        'attn': 1,
        'brnn': 1,
        'decay': 45,
        'emb': 128,
        'rnn': 200,
        'dropout': 0.5,
    },
}

def wrap(b):
    b = torch.stack(b, 0).t().contiguous()
    if USE_CUDA:
        b = b.cuda()
    # b = Variable(b, volatile=self.eval)
    return b

def _batchify(data, align_right=False, include_lengths=False, pad_id=None):
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

def _get_single_item(index, batch_size, src, tgt, idx, pad_id):
    src_batch, src_lengths = _batchify([src[index]] * batch_size, include_lengths=True, pad_id=pad_id)
    tgt_batch = _batchify([tgt[index]] * batch_size, pad_id=pad_id)
    idx_batch = tuple([idx[index]] * batch_size)
    indices = range(batch_size)

    batch = zip(indices, src_batch, tgt_batch, idx_batch)
    batch, src_lengths = zip(*sorted(zip(batch, src_lengths), key=lambda x: -x[1]))
    indices, src_batch, tgt_batch, idx_batch = zip(*batch)

    return (wrap(src_batch), list(src_lengths)), None, wrap(tgt_batch), indices, None, idx_batch

def _predict(model, batch):
    outputs = model(batch, True)
    logits = model.generator.forward(outputs)
    return logits.data.max(1)[1].view(outputs.size(0), -1).T

def _encode(tokens, vocab, unk_id=None):
    try:
        if len(tokens) > TOKEN_LIST_SIZE:
            tokens = tokens[:TOKEN_LIST_SIZE]
        return list(map(lambda x: vocab.get(x, unk_id), tokens))
    except Exception as e:
        logging.error(e)

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
    model_values = '_'.join([f'{k}{v}' for k, v in model_conf.items()])
    model_id = MODEL_BASE_ID.format(model_values)
    model_name = os.path.join(MODEL_BASE_DIR, model_id, f'{model_id}_{MODEL_NUMBER[LANG]}.pt')
    logging.debug(f'model: {model_name}')
    checkpoint = torch.load(model_name)
    model = checkpoint['model']

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

def _load_tokens(data_conf, data_set_name, code_vocab, qt_vocab, unk_id, pad_id, eos_id):
    code_tokens_set, qt_tokens_set, split_indices = [_load_file(os.path.join(WORK_DIR, filename))
                                                     for _, filename in data_conf.items()]

    _, qt_ids, code_ids = zip(*split_indices[data_set_name])

    code_tokens = list(map(code_tokens_set.get, code_ids))
    code_indices = list(map(lambda e: _set_eos_and_pad(_encode(e, code_vocab), pad_id), code_tokens))
    code_tensors = _get_tensors(code_indices)

    qt_tokens = list(map(qt_tokens_set.get, qt_ids))
    qt_indices = list(map(lambda e: _set_eos_and_pad(_encode(e, qt_vocab, unk_id), pad_id, eos_id), qt_tokens))
    qt_tensors = _get_tensors(qt_indices)

    return code_tokens, code_indices, code_tensors, qt_tokens, qt_indices, qt_tensors, code_ids

if __name__ == '__main__':
    _set_seed(SEED)

    code_vocab, code_vocab_ids, qt_vocab, qt_vocab_ids = _load_vocabs(VOCAB_CONF)
    unk_id = qt_vocab.get('<unk>')
    pad_id = qt_vocab.get('<pad>')
    eos_id = qt_vocab.get('<eos>')

    code_tokens, code_indices, code_tensors, qt_tokens, qt_indices, qt_tensors, code_ids =\
        _load_tokens(DATA_CONF, 'train', code_vocab, qt_vocab, unk_id, pad_id, eos_id)

    model = _load_model(MODEL_CONF[LANG])

    for n, index in enumerate(range(N_PREDS)):
        batch = _get_single_item(index, BATCH_SIZE, code_tensors, qt_tensors, code_ids, pad_id)
        preds = _predict(model, batch)
        pred = _decode(preds[0].tolist(), qt_vocab_ids, eos_id)
        print(f'PREDICTION {n}\nCODE: {code_tokens[index]}\nQT TRUE: {qt_tokens[index]}\nQT PREDICTION: {pred}')
