"""
@author: Iñigo Martínez <inigo.martinez@tecnalia.com>
@copyright: 2022 Tecnalia
"""
import pdb
from collections import ChainMap
from datasets import load_dataset
from itertools import chain, cycle, groupby, product
from data import CodennDataset, StaQCDataset
from models import *
import numpy as np
import os
from pathlib import Path
import pickle
import random
import sys
import torch
from torch import optim
from utils import ACC, MAP, MRR, NDCG

'''
- tenemos un qt
  - tokenizarla (qt_vocab)
  - obtener la representación del modelo entrenado

- coger todos los códigos
  - tokenizar (code_vocab)
  - obtener la representación del modelo entrenado

- buscar la similaridad de coseno del qt respecto a todos los códigos y sacar un ranking

- el mejor ranking es el código correspondiente

- decodificar los tokens del código
'''

MODEL_FILE = '../checkpoint/QC_valcodenn/qtlen_20_codelen_120_qtnwords_176946_codenwords_1907029_batch_256_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/best_model.ckpt'

CONF = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    '<unk>': 3,

    'workdir': '../data/',

    'qt_n_words': 176946,
    'code_n_words': 1907029,

    'emb_size': 200,
    'lstm_dims': 400,

    'use_anno': False,

    'lr': 0.001,

    'bow_dropout': 0.25,
    'seqenc_dropout': 0.25,
    'margin': 0.05,

    'code_encoder': 'bilstm',

    'qt_len': 20,
    'code_len': 120,
    'anno_len': 20,

    'val_code': 'codenn_combine_new.java.dev.code.pkl',

    'val_qt': 'codenn_combine_new.java.dev.qt.pkl',
    'val_anno': 'codenn.java.dev.ga.pkl',
    'test_qt': 'codenn_combine_new.java.eval.qt.pkl',
    'test_anno': 'codenn.java.eval.ga.pkl',

    'vocab_qt': 'java.qt.vocab.pkl',
    'vocab_code': 'java.code.vocab.pkl',
    'vocab_anno': 'java.qt.vocab.pkl',
}

USE_CUDA = False

def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if USE_CUDA:
        tensor = tensor.cuda()
    return tensor


class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['workdir']

    #######################
    # Model Loading / saving #####
    #######################
    def save_model(self, model):
        if not os.path.exists(self.conf['model_directory']):
            os.makedirs(self.conf['model_directory'])
        torch.save(model.state_dict(), self.conf['model_directory'] + 'best_model.ckpt')

    def load_model(self, model):
        assert os.path.exists(self.conf['model_directory'] + 'best_model.ckpt'), 'Weights for saved model not found'
        model.load_state_dict(torch.load(self.conf['model_directory'] + 'best_model.ckpt'))


    #######################
    # Training #####
    #######################
    def train(self, model, val_setup="staqc"):
        """
        Trains an initialized model
        :param model: Initialized model
        :return: None
        """
        optimizer = optim.Adam(model.parameters(), lr=self.conf['lr'])

        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        max_patience = self.conf['patience']

        train_set = StaQCDataset(self.path, self.conf, "train")
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                  shuffle=True, drop_last=True, num_workers=1)

        # val set
        if val_setup == "staqc":
            val = StaQCDataset(self.path, self.conf, "val")
        elif val_setup == "codenn":
            val = CodennDataset(self.path, self.conf, "val")
        else:
            raise Exception("Invalid val_setup %s!" % val_setup)

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            if val_setup == "codenn":
                _, max_mrr, _, _ = self.eval_codenn(model, 50, val)
            else:
                _, max_mrr, _, _ = self.eval(model, 50, val)
        else:
            max_mrr = -1

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses = []

            model = model.train()

            for qts, good_cands, bad_cands in data_loader:
                qts, good_cands, bad_cands = gVar(qts), gVar(good_cands), gVar(bad_cands)

                loss, good_scores, bad_scores = model(qts, good_cands, bad_cands)

                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every == 0:
                    print('epo:[%d/%d] itr:%d Loss=%.5f' % (epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []
                itr = itr + 1

            if epoch % valid_every == 0:
                print("validating..")
                if val_setup == "codenn":
                    print("val_setup: codenn")
                    acc1, mrr, map, ndcg = self.eval_codenn(model, 50, val)
                else:
                    acc1, mrr, map, ndcg = self.eval(model, 50, val)

                if mrr > max_mrr:
                    self.save_model(model)
                    patience = 0
                    print("Model improved. Saved model at %d epoch" % epoch)
                    max_mrr = mrr
                else:
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1

            if patience >= max_patience:
                print("Patience Limit Reached. Stopping Training")
                break

    ########################
    # Evaluation on CodeNN #
    ########################
    def eval_codenn(self, model, poolsize, dataset, bool_collect=False):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  num_workers=1)
        model = model.eval()

        sims_collection = []
        accs, mrrs, maps, ndcgs = [], [], [], []
        for qts, cands in data_loader:
            cands = gVar(cands)

            cands_repr = model.cand_encoding(cands)

            if isinstance(qts, list):
                assert len(qts) == 3
                qts = [gVar(qts_i) for qts_i in qts]
            else:
                qts = [gVar(qts)]

            sims_per_qts = []
            for qts_i in qts:
                qt_repr = model.qt_encoding(qts_i)

                sims = model.scoring(qt_repr, cands_repr).data.cpu().numpy()
                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                predict = [int(k) for k in predict]
                real = [0]  # index of the positive sample

                # save
                sims_per_qts.append(sims)

                mrrs.append(MRR(real, predict))
                accs.append(ACC(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))

            sims_collection.append(sims_per_qts)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_codenn_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(mrrs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    #######################
    # Evaluation on StaQC #####
    #######################
    def eval(self, model, poolsize, dataset, bool_collect=False):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize,
                                                  shuffle=False, drop_last=True,
                                                  num_workers=1)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []

        sims_collection = []
        for qts, cands, _ in data_loader:
            qts, cands = gVar(qts), gVar(cands)
            qts_repr = model.qt_encoding(qts)
            cands_repr = model.cand_encoding(cands)

            _poolsize = len(qts) if bool_collect else poolsize # true poolsize
            for i in range(_poolsize):
                _qts_repr = qts_repr[i].expand(_poolsize, -1)

                scores = model.scoring(_qts_repr, cands_repr).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                real = [i]  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
                sims_collection.append(scores)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset.data_name)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


def _decode(indices, vocab):
    return list(map(vocab.get, indices))


def _invert(d):
    return {val: key for (key, val) in d.items()}


def _format_str_list(text, c):
    return ' '.join(text[:text.index(c)] if c in text else text)


if __name__ == '__main__':
    # searcher = CodeSearcher(CONF)
    with open(os.path.join(CONF['workdir'], CONF['vocab_code']), 'rb') as f:
        vocab_code = _invert(pickle.load(f, encoding='latin1'))

    with open(os.path.join(CONF['workdir'], CONF['vocab_qt']), 'rb') as f:
        vocab_qt = _invert(pickle.load(f, encoding='latin1'))

    model = JointEmbeder(CONF)
    model.load_state_dict(torch.load(MODEL_FILE))

    val = CodennDataset(CONF['workdir'], CONF, "val")
    data_loader = torch.utils.data.DataLoader(dataset=val, batch_size=50, shuffle=False, num_workers=1)

    for qts, cands in data_loader:
        cands = gVar(cands)

        cands_repr = model.cand_encoding(cands)

        if isinstance(qts, list):
            assert len(qts) == 3
            qts = [gVar(qts_i) for qts_i in qts]
        else:
            qts = [gVar(qts)]

        sims_per_qts = []
        for qts_i in qts:
            qt_repr = model.qt_encoding(qts_i)

            sims = model.scoring(qt_repr, cands_repr).data.cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict = list(map(int, predict))

            qt = _decode(qts_i[0].numpy().tolist(), vocab_qt)
            qt = _format_str_list(qt, '<pad>')
            code = _decode(cands[predict.index(0)].numpy().tolist(), vocab_code)
            code = _format_str_list(code, ',')
            print(f'QT: {qt}\nCODE: {code}')
            # pdb.set_trace()
            '''
            # save
            sims_per_qts.append(sims)

            mrrs.append(MRR(real, predict))
            accs.append(ACC(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
            '''
    print('x')
