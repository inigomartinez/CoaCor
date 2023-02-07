"""
@author: Iñigo Martínez <inigo.martinez@tecnalia.com>
@copyright: 2022-2023 Tecnalia
"""

from ast import literal_eval
from collections import ChainMap
from datasets import load_dataset
from itertools import chain, groupby, product
import os
import pandas as pd
from pathlib import Path
import pickle
import random
from sklearn.model_selection import train_test_split

# HARDCODED FROM Contants.py
BASE_VOCAB = {
    'qt': {
        0: '<pad>',
        1: '<sos>',
        2: '<eos>',
        3: '<unk>',
    },
    'code': {},
}

SOURCE_DATA_DIR = 'data/source/'
PKL_FILES = [
    'split_indices_{}_cleaned.pkl',
    '{}_index_to_tokenized_code.pkl',
    '{}_index_to_tokenized_qt.pkl',
    'staqc_idx2gen_token.pkl',
]

VOCAB_FILES = [

    '{}.{}.vocab.pkl',
]

CODE_RETRIEVAL_DATA_DIR = 'code/CodeRetrieval-Main/data/'

CODENN_DATA_DIR = 'codenn_{}'
CODENN_FILES = {
    'codenn.{}.ix_to_tokenized_code.pkl',
    'codenn.{}.ix_to_tokenized_qt.pkl',
    'codenn.{}.qid_cid_pair.gen.dataset.pkl',
}

CODENN_ITEMS = 50
CODENN_EVAL_RESULT = [True] + ([False] * (CODENN_ITEMS - 1))

CR_DATA_DIR = 'code/code_annotation/dataset/cr_data/'

LANG = 'typescript'

TRAIN_TYPE = ['train', 'test', 'val']
SET_TYPES = ['qt', 'code']
DEV_TYPES = ['dev', 'eval']

# DEV_SET_SIZES = 10000
DEV_SET_SIZES = [111, 100]
DEV_STEP_POS = 3
DEV_ELEMENTS = 50
# (20 runs as Iyer et al.)
DEV_RUNS = 20
DEV_ANNO_TYPES = ['rl_bleu', 'rl_mrr', 'sl']

OUTPUT_DIR = 'data'
DATASET_NAME = 'proxya'
# DATASET_NAME = 'code_search_net'

PROXYA_DATASET_SRC_DIR = os.path.join(OUTPUT_DIR, DATASET_NAME, 'snippets')

PROXYA_DATASET_FILES = {
    'java': [
        'ctd-servicios-de-apoyo-a-la-gestion-tic-prorroga-src_java.csv',
        'educamosclm-arq_java.csv',
        'reutilizacion_factoria_java.csv',
    ],
    'typescript': [
        'ai-management_typescript.csv',
        'alerts-management_typescript.csv',
        'dashboards_typescript.csv',
        'device-management_typescript.csv',
        'historical-management_typescript.csv',
        'issues-management_typescript.csv',
        'kpi-management_typescript.csv',
        'lib-commons-node_typescript.csv',
        'report-management_typescript.csv',
        'rules-management_typescript.csv',
        'runtime-management_typescript.csv',
        'scheduler_typescript.csv',
        'smartboards-management_typescript.csv',
        'menu-management_typescript.csv',
        'user-management_typescript.csv',
    ]
}

CHECK = False

current_token_id = 1


def _read(filename):
    """
    Parameters
    ----------
    filename : str
        the file name to be loaded

    Returns
    -------
    pickle
        the pickle representation
    """
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def _write(filename, o):
    """
    Parameters
    ----------
    filename : str
        the file name to be loaded
    o : object
        the data to be written in the file
    """
    filename = os.path.join(OUTPUT_DIR, DATASET_NAME, LANG, filename)
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)

    with open(filename, 'wb') as f:
        pickle.dump(o, f, protocol=2)


def _invert(d):
    """
    Parameters
    ----------
    d : dict
        the data list

    Returns
    -------
    dict
        the inverted dictionary
    """
    return {val: key for (key, val) in d.items()}


def _encode(tokens, vocab, unk=None):
    """
    Parameters
    ----------
    tokens : list
        the list of tokens to be encoded
    vocab : dict
        the vocabulary with the tokens <-> indices relationship
    unk : int
        the identifier to be used as unk if any token has not a known identifier

    Returns
    -------
    str
        the string of the tokens represented by the correspondent indices
    """
    try:
        return ' '.join(map(str, map(lambda x: vocab.get(x, unk), tokens)))
    except Exception as e:
        print(e)


def _encode_tokens(tokens_data, index_vocabs):
    """
    Parameters
    ----------
    tokens_data : list
        the list of tokens to be encoded
    index_vocabs : dict
        the vocabulary with the tokens <-> indices relationship

    Returns
    -------
    typle
        the tokens represented by the correspondent indices and the tokens data
    """
    return tuple(map(lambda e: (e, _encode(e, index_vocabs)), tokens_data))


def _decode(indices, vocab):
    """
    Parameters
    ----------
    indices : str
        the indices of a token list
    vocab : dict
        the vocabulary with the indices <-> tokens relationship

    Returns
    -------
    list
        the tokens represented by the correspondent indices
    """
    return [vocab[idx] for idx in map(int, indices.split())]


def _id_tokens(tokens, start=0, sort=False):
    """
    Parameters
    ----------
    tokens : list
        the list of tokens to be assigned and unique identifier
    start : int
        the starting identifier
    sort : bool
        if the list of tokens should be sorted

    Returns
    -------
    list
        the list of tokens with the assigned identifiers
    """
    if sort:
        tokens = sorted(tokens)
    return dict(zip(range(start, start + len(tokens)), tokens))


def _id_global_tokens(tokens, sort=False):
    """
    Parameters
    ----------
    tokens : list
        the list of tokens to be assigned and unique identifier
    sort : bool
        if the list of tokens should be sorted

    Returns
    -------
    list
        the list of tokens with the assigned identifiers using the global identifier list
    """
    global current_token_id

    if sort:
        tokens = sorted(tokens)

    end_id = current_token_id + len(tokens)
    id_tokens = dict(zip(range(current_token_id, end_id), tokens))
    current_token_id = end_id

    return id_tokens


def _process_data(doc_tokens, code_tokens, limit=None):
    """
    Parameters
    ----------
    doc_tokens : list
        the list of annotation tokens to be processed
    code_tokens : list
        the list of source code tokens to be processed
    limit : int
        if the token list must be trimmed to the limited value

    Returns
    -------
    dict
        the list of tokens with their particular vocabulary
    """
    tokens = {'qt': doc_tokens, 'code': code_tokens}

    if limit:
        tokens = {tokens_set_type: list(map(lambda t: t[:limit], tokens_set_data))
                  for tokens_set_type, tokens_set_data in tokens.items()}

    vocabs = {k: set(chain(*v)) for k, v in tokens.items()}

    return {'tokens': tokens, 'vocabs': vocabs}


def _get_index_set_type_vocabs(data, sort=False):
    """
    Parameters
    ----------
    data : dict
        the different vocabularies data
    sort : bool
        if the list of tokens should be sorted

    Returns
    -------
    dict
        the list of indexed vocabularies sorted by set type
    """
    id_set_type_vocabs = {set_type: BASE_VOCAB[set_type] | _id_tokens(set(chain(*[vocabs_set_data
                                                                                  for tokens_data in data.values()
                                                                                  for vocabs_set_type, vocabs_set_data
                                                                                  in tokens_data['vocabs'].items()
                                                                                  if set_type == vocabs_set_type])),
                                                                      # FIXME: UGLY HACK to set starting value
                                                                      start=max(BASE_VOCAB[set_type].keys(),
                                                                                default=-1) + 1,
                                                                      sort=sort)
                          for set_type in SET_TYPES}

    return {vocabs_set_type: _invert(id_vocabs_set_data)
            for vocabs_set_type, id_vocabs_set_data in id_set_type_vocabs.items()}


def _get_split_indices(data):
    """
    Parameters
    ----------
    data : dict
        the different data items

    Returns
    -------
    dict
        the list of indexed data sorted by set type
    """
    split_indices = {tokens_type: tuple(map(lambda e: (e[0], e[0], e[2]), tokens_data))
                     for tokens_type, tokens_data in data.items()}
    split_indices['valid'] = split_indices.pop('validation')

    return split_indices


def _get_indices_and_tokens(id_data):
    """
    Parameters
    ----------
    id_data : dict
        the different data items

    Returns
    -------
    dict
        the list of indexed data
    """
    return dict(map(lambda e: (e[0], list(map(lambda l: tuple(chain(*l)),
                                              zip(e[1]['qt'].items(), e[1]['code'].items())))),
                    id_data.items()))


def _process_tokens(data, index_set_type_vocabs):
    """
    Parameters
    ----------
    data : dict
        the different data items
    index_set_type_vocabs : dict
        the different set type dictionaries with indices

    Returns
    -------
    dict
        the list of indexed data
    """
    return {tokens_type: {tokens_set_type: _id_global_tokens(_encode_tokens(tokens_set_data,
                                                                            index_set_type_vocabs[tokens_set_type]))
                          for tokens_set_type, tokens_set_data in tokens_data['tokens'].items()}
            for tokens_type, tokens_data in data.items()}


def _write_vocabs(lang, index_set_type_vocabs):
    """
    Parameters
    ----------
    lang : str
        the correspondent language
    index_set_type_vocabs : dict
        the different set type vocabularies with indices
    """
    for set_type, set_data in index_set_type_vocabs.items():
        name = f'{lang}.{set_type}.vocab.pkl'
        _write(os.path.join(SOURCE_DATA_DIR, name), set_data)
        _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, name), set_data)
        _write(os.path.join(CR_DATA_DIR, name), set_data)


def _write_index_to_tokenized_data(lang, id_data):
    """
    Parameters
    ----------
    lang : str
        the correspondent language
    id_data : dict
        the different set type tokens data with indices
    """
    set_type_index_to_tokenized = {set_type: dict(ChainMap(*[dict(map(lambda e: (e[0], e[1][0]),
                                                                      tokens_set_data.items()))
                                                             for tokens_data in id_data.values()
                                                             for tokens_set_type, tokens_set_data in tokens_data.items()
                                                             if tokens_set_type == set_type]))
                                   for set_type in SET_TYPES}

    for set_type, set_data in set_type_index_to_tokenized.items():
        _write(os.path.join(SOURCE_DATA_DIR, f'{lang}_index_to_tokenized_{set_type}.pkl'), set_data)


def _write_indexed_encoded_data(lang, encoded_data):
    """
    Parameters
    ----------
    lang : str
        the correspondent language
    encoded_data : dict
        the different set type encoded tokens data
    """
    for tokens_type, tokens_data in encoded_data.items():
        if tokens_type == 'validation':
            tokens_type = 'val'

        for tokens_set_type, tokens_set_data in tokens_data.items():
            name = f'{lang}.{tokens_type}.{tokens_set_type}.pkl'
            data = list(map(lambda k, v: (k, v[1]), tokens_set_data.keys(), tokens_set_data.values()))
            _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, name), data)

            if tokens_type == 'train':
                _write(os.path.join(CR_DATA_DIR, name), data)


def _write_codenn_dev_files(lang, set_data):
    """
    Parameters
    ----------
    lang : str
        the correspondent language
    set_data : dict
        the different set type set tokens data
    """
    out_dir = os.path.join(SOURCE_DATA_DIR, CODENN_DATA_DIR.format(lang))

    # we limit the available split indices
    set_data['val'] = set_data.pop('validation')
    for anno_type in DEV_ANNO_TYPES:
        for train_type, train_data in set_data.items():
            _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'{lang}.{train_type}.anno_{anno_type}.pkl'),
                   list(map(lambda e: e[1][1], train_data)))

    set_data = [random.sample(set_data['train'], size) for size in DEV_SET_SIZES]

    for dev_type, indices_and_tokens in zip(DEV_TYPES, set_data):
        # in documentation (aka qt) we don't care about different IDs
        # id start at 0 in dev/eval sets, so we substract 1 to the ID
        qt_ids, qt_data, code_ids, code_data = zip(*indices_and_tokens)

        qt_tokens, qt_encoded = zip(*qt_data)
        code_tokens, code_encoded = zip(*code_data)

        _write(os.path.join(out_dir, f'codenn.{dev_type}.qid_cid_pair.gen.dataset.pkl'), list(zip(qt_ids, code_ids)))
        for set_type, ids, tokens in zip(SET_TYPES, [qt_ids, code_ids], [qt_tokens, code_tokens]):
            _write(os.path.join(out_dir, f'codenn.{dev_type}.ix_to_tokenized_{set_type}.pkl'), dict(zip(ids, tokens)))

        combine_data = list(chain(*chain(*[[[([qt_set] * DEV_STEP_POS, code)] +
                                            list(product([[qt_set] * DEV_STEP_POS],
                                                         random.sample(code_encoded, DEV_ELEMENTS - 1)))
                                            for qt_set, code in zip(qt_encoded, code_encoded)]
                                           for _ in range(DEV_RUNS)])))

        for set_type, set_data in zip(SET_TYPES, zip(*combine_data)):
            _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'codenn_combine_new.{lang}.{dev_type}.{set_type}.pkl'),
                   list(set_data))

        for anno_type in DEV_ANNO_TYPES:
            dev_data, _ = zip(*combine_data)
            dev_data, _, _ = zip(*dev_data)
            _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'codenn_combine_new.{lang}.{dev_type}.anno_{anno_type}.pkl'),
                   random.sample(dev_data, k=len(dev_data)))

        _write(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'codenn.{lang}.{dev_type}.ga.pkl'), dict(zip(qt_ids, qt_encoded)))


def _write_files(data, lang, sort=False):
    """
    Parameters
    ----------
    data : dict
        the tokens data
    lang : str
        the correspondent language
    sort : bool
        if the list of tokens should be sorted
    """
    index_set_type_vocabs = _get_index_set_type_vocabs(data, sort)
    _write_vocabs(lang, index_set_type_vocabs)

    id_data = _process_tokens(data, index_set_type_vocabs)
    del data

    _write_index_to_tokenized_data(lang, id_data)
    _write_indexed_encoded_data(lang, id_data)

    indices_and_tokens = _get_indices_and_tokens(id_data)
    del id_data

    _write(os.path.join(SOURCE_DATA_DIR, f'split_indices_{lang}_cleaned.pkl'), _get_split_indices(indices_and_tokens))

    _write_codenn_dev_files(lang, indices_and_tokens)


def _load_all(lang):
    """
    Parameters
    ----------
    lang : str
        the correspondent language

    Returns
    -------
    dict
        the annotation and source code data
    dict
        the annotation vocabulary
    dict
        the source code vocabulary
    dict
        the development files data
    """
    doc_datas = {Path(filename.format(lang)).stem: _read(os.path.join(SOURCE_DATA_DIR, filename.format(lang)))
                 for filename in PKL_FILES}

    codes_datas = {lang: {f'{lang}.{train_type}.{set_type}':
                              _read(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'{lang}.{train_type}.{set_type}.pkl'))
                          for train_type in TRAIN_TYPE
                          for set_type in SET_TYPES}
                   for lang in langs}

    set_code = list(chain(*[codes_datas[f'{lang}.{set_type}.code'] for set_type in ['train', 'test', 'val']]))

    doc_vocab_token_index = doc_datas[f'{lang}.qt.vocab']
    code_vocab_token_index = doc_datas[f'{lang}.code.vocab']
    doc_vocab = _invert(doc_datas[f'{lang}.qt.vocab'])
    code_vocab = _invert(doc_datas[f'{lang}.code.vocab'])

    dev_datas = {Path(filename.format(dev_type)).stem: _read(os.path.join(SOURCE_DATA_DIR,
                                                                          CODENN_DATA_DIR.format(lang),
                                                                          filename.format(dev_type)))
                 for dev_type in DEV_TYPES
                 for filename in CODENN_FILES}

    codenn_ga_datas = {Path(f'codenn.{lang}.{dev_type}.ga.pkl').stem:
                           _read(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'codenn.{lang}.{dev_type}.ga.pkl'))
                       for dev_type in DEV_TYPES}

    codenn_combine_datas = {Path(f'codenn_combine_new.{lang}.{dev_type}.{set_type}.pkl').stem:
                                _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                                   f'codenn_combine_new.{lang}.{dev_type}.{set_type}.pkl'))
                            for dev_type in DEV_TYPES
                            for set_type in SET_TYPES}

    anno_datas = {Path(f'{lang}.{train_type}.anno_{anno_type}.pkl').stem:
                      _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                         f'{lang}.{train_type}.anno_{anno_type}.pkl'))
                  for train_type in TRAIN_TYPE
                  for anno_type in DEV_ANNO_TYPES}

    codenn_combine_anno_datas = {Path(f'codenn_combine_new.{lang}.{dev_type}.anno_{anno_type}.pkl').stem:
                                     _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                                        f'codenn_combine_new.{lang}.{dev_type}.anno_{anno_type}.pkl'))
                                 for dev_type in DEV_TYPES
                                 # 'codenn_gen' anno_type/qn_mode is not used
                                 for anno_type in DEV_ANNO_TYPES}

    codenn_eval_code_token = dict(dev_datas['codenn.eval.ix_to_tokenized_code'])
    codenn_eval_qt_token = dict(dev_datas['codenn.eval.ix_to_tokenized_qt'])
    codenn_dev_code_token = dict(dev_datas['codenn.dev.ix_to_tokenized_code'])
    codenn_dev_qt_token = dict(dev_datas['codenn.dev.ix_to_tokenized_qt'])
    codenn_dev_ga = dict(codenn_ga_datas['codenn.sql.dev.ga'])
    codenn_eval_ga = dict(codenn_ga_datas['codenn.sql.eval.ga'])

    combine_new_dev_qt = codenn_combine_datas[f'codenn_combine_new.{lang}.dev.qt']
    combine_new_dev_qt_single = list(k for k, _ in groupby(combine_new_dev_qt))
    combine_new_dev_code = codenn_combine_datas[f'codenn_combine_new.{lang}.dev.code']

    codenn_dev_code_encoded = {idx: _encode(tokens, code_vocab_token_index)
                               for idx, tokens in codenn_dev_code_token.items()}

    codenn_dev_qt_encoded = {idx: _encode(tokens, doc_vocab_token_index, doc_vocab_token_index['<unk>'])
                             for idx, tokens in codenn_dev_qt_token.items()}

    combine_new_dev_pairs = list(zip(combine_new_dev_qt, combine_new_dev_code))

    combine_new_dev_pairs_decoded = [([_decode(qt, doc_vocab) for qt in qts],
                                      _decode(code, code_vocab))
                                     for qts, code in combine_new_dev_pairs]

    combine_new_eval = [list(zip(combine_new_dev_pairs[n: n + CODENN_ITEMS - 1], CODENN_EVAL_RESULT))
                        for n in range(0, len(combine_new_dev_pairs), CODENN_ITEMS)]

    # return doc_datas, codes_datas, doc_vocab, code_vocab, dev_datas

    return doc_datas, doc_vocab, code_vocab, dev_datas


def _check_datas(langs):
    """
    Parameters
    ----------
    langs : list
        the languages
    """
    doc_datas = {lang: {Path(filename.format(lang)).stem: _read(os.path.join(SOURCE_DATA_DIR, filename.format(lang)))
                        for filename in PKL_FILES}
                 for lang in langs}

    return None


def _check_train(langs):
    """
    Parameters
    ----------
    langs : list
        the languages
    """
    train_datas = {lang: {f'{lang}.{train_type}.{set_type}':
                              _read(os.path.join(CODE_RETRIEVAL_DATA_DIR, f'{lang}.{train_type}.{set_type}.pkl'))
                          for train_type in TRAIN_TYPE
                          for set_type in SET_TYPES}
                   for lang in langs}

    return None


def _check_anno(langs):
    """
    Parameters
    ----------
    langs : list
        the languages
    """
    anno_datas = {lang: {Path(f'{lang}.{train_type}.anno_{anno_type}.pkl').stem:
                             _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                                f'{lang}.{train_type}.anno_{anno_type}.pkl'))
                         for train_type in TRAIN_TYPE
                         for anno_type in DEV_ANNO_TYPES}
                  for lang in langs}

    '''
    for n in range(10):
        print(f'{n} bleu: {_decode(sql_train_anno_rl_bleu[n], qt_vocab_token_index)}\n'
              f'{n} mrr: {_decode(sql_train_anno_rl_mrr[n], qt_vocab_token_index)}\n'
              f'{n} sl: {_decode(sql_train_anno_sl[0], qt_vocab_token_index)}')
    '''

    codenn_combine_anno_datas = {lang: {Path(f'codenn_combine_new.{lang}.{dev_type}.anno_{anno_type}.pkl').stem:
                                            _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                                               f'codenn_combine_new.{lang}.{dev_type}.anno_{anno_type}.pkl'))
                                        for dev_type in DEV_TYPES
                                        # 'codenn_gen' anno_type/qn_mode is not used
                                        for anno_type in DEV_ANNO_TYPES}
                                 for lang in langs}

    return None


def _check_combine(langs):
    """
    Parameters
    ----------
    langs : list
        the languages
    """
    codenn_combine_datas = {lang: {Path(f'codenn_combine_new.{lang}.{dev_type}.{set_type}.pkl').stem:
                                       _read(os.path.join(CODE_RETRIEVAL_DATA_DIR,
                                                          f'codenn_combine_new.{lang}.{dev_type}.{set_type}.pkl'))
                                   for dev_type in DEV_TYPES
                                   for set_type in SET_TYPES}
                            for lang in langs}

    return None


def _load_dataset_code_search_net(lang):
    """
    Parameters
    ----------
    lang : str
        the correspondant language

    Returns
    -------
    dict
        the code search dataset data
    """
    dataset = load_dataset('code_search_net', lang)
    return {k: _process_data(v['func_documentation_tokens'], v['func_code_tokens']) for k, v in dataset.items()}


def _load_dataset_proxya(lang):
    """
    Parameters
    ----------
    lang : str
        the correspondant language

    Returns
    -------
    dict
        the proxya dataset data
    """
    data = pd.concat([pd.read_csv(os.path.join(PROXYA_DATASET_SRC_DIR, f),
                                  converters={'docstring_tokens': literal_eval, 'func_code_tokens': literal_eval})
                      for f in PROXYA_DATASET_FILES[lang]], axis=0)
    data = data[data['docstring_tokens'].astype(bool) & data['func_code_tokens'].astype(bool)]

    qt_train, qt_test, code_train, code_test = train_test_split(data['docstring_tokens'].to_list(),
                                                                data['func_code_tokens'].to_list(),
                                                                test_size=0.2, random_state=1)

    qt_train, qt_val, code_train, code_val = train_test_split(qt_train, code_train, test_size=0.25, random_state=1)

    data = {
        'train': {'qt': qt_train, 'code': code_train},
        'test': {'qt': qt_test, 'code': code_test},
        'validation': {'qt': qt_val, 'code': code_val},
    }

    return {k: _process_data(v['qt'], v['code']) for k, v in data.items()}


if __name__ == '__main__':
    CHECK = False
    if CHECK:
        langs = ['sql', 'java']

        vocabs = {lang: {set_type: _invert(_read(os.path.join(SOURCE_DATA_DIR, f'{lang}.{set_type}.vocab.pkl')))
                         for set_type in SET_TYPES}
                  for lang in langs}

        # datas = {lang: _load_all(lang) for lang in ['sql']}
        # datas = {lang: _load_all(lang) for lang in langs}

        # _check_datas(langs)
        _check_train(langs)
        _check_anno(langs)
        _check_combine(langs)

    if DATASET_NAME == 'proxya':
        data = _load_dataset_proxya(LANG)
    else:
        # CodeSearchNet
        data = _load_dataset_code_search_net(LANG)

    _write_files(data, LANG, sort=True)
