import gzip
import numpy as np
import ujson
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import random
import pickle
import csv
import os

NUM_PROCESSORS = 12
UNK = '<UNK>'


def check_forbidden(token):
    for s in ('|', ':', '\n'):
        if s in token:
            return False
    return True


def make_contexts(tokens, width=5, dictionary=None):
    word2contexts = defaultdict(list)

    for paragraph in tokens:

        paragraph = [_ for _ in paragraph if check_forbidden(_)]

        for cw_index in range(len(paragraph)):
            left_index = max(0, cw_index - width)
            right_index = min(len(paragraph), cw_index + width)
            context = paragraph[left_index:cw_index] + paragraph[
                                                       cw_index + 1:right_index]
            if dictionary is not None:
                context = [dictionary.get(c, dictionary[UNK])
                           for c in context]
            word2contexts[paragraph[cw_index]].append(context)
    return word2contexts


def sum_contexts(cs1, cs2, inplace=True):
    if inplace:
        res = cs1
    else:
        res = cs1.copy()
    for k in cs2.keys():
        if k in res:
            res[k].extend(cs2[k])
        else:
            res[k] = cs2[k]
    return res


def default_pipe(data_gzip='../wiki150k',
                 size=1000,
                 dump_flag=True,
                 dump_folder=None):
    with gzip.open(data_gzip, 'r') as input:
        data = []
        i = 0
        for _ in tqdm(input):
            _json = ujson.loads(_)
            data.append(_json)
            i += 1
            if i == size: break

    pool = Pool(NUM_PROCESSORS)
    all_contexts = list(tqdm(pool.map(make_contexts,
                                      tqdm([(_['content_moses_lemmas'])
                                            for _ in tqdm(data)]))))
    pool.close()
    del pool

    res = defaultdict(list)
    for c in tqdm(all_contexts):
        sum_contexts(res, c, inplace=True)

    selected_items = []
    for k, v in tqdm(res.items()):
        if 10 < len(v) < 600:
            selected_items.append((k, v))
    del res

    random.shuffle(selected_items)

    # 1. built dictionary: tf, df for each word
    # 2. flatten dataset: only contexts, without key words
    docs_count = 0
    dictionary_tf = Counter()
    dictionary_df = Counter()
    for k, v in selected_items:
        for d in v:
            # docs.extend(d)
            dictionary_tf.update(Counter(d))
            dictionary_df.update(Counter(set(d)))
            docs_count += 1
    # del selected_items

    dictionary_df_rate = {w: d / docs_count for w, d in dictionary_df.items()}
    # filter by min_df=45, max_df_rate=.02
    min_df_selected = {w for w, d in dictionary_df.items() if d >= 45}
    max_df_rate_selected = {w for w, d in dictionary_df_rate.items()
                            if d <= 0.02}
    dictionary = min_df_selected & max_df_rate_selected
    dictionary = {w: i for i, w in enumerate(dictionary)}
    dictionary.setdefault(UNK, len(dictionary))

    word_inx_docs = []
    empty_docs_count = 0
    for k, docs in selected_items:
        for doc in docs:
            _doc = []
            for w in doc:
                _doc.append(
                    dictionary.get(w, dictionary[UNK])
                )

            if len([i for i in _doc if i != dictionary[UNK]]) == 0:
                empty_docs_count += 1
            else:
                word_inx_docs.append(_doc)
    del selected_items

    if dump_flag is True:
        if dump_folder is None or not os.path.exists(dump_folder):
            dump_folder = '/tmp'
        pickle.dump(word_inx_docs, open(os.path.join(dump_folder,
                                                     'word_inx_docs_2.pickle',
                                                     'wb')))
        pickle.dump(dictionary, open(os.path.join(dump_folder,
                                                  'dictionary.pickle', 'wb')))

    print('Len of word_inx_docs: {}'.format(len(word_inx_docs)))
    print('Empty_docs count: {}'.format(empty_docs_count))
    return word_inx_docs, dictionary


def bigartm_dict_pipe(artm_dict_path=None,
                      data_gzip='../wiki150k',
                      size=1000,
                      dump_flag=True,
                      dump_folder=None):
    dictionary = dict()
    with open(artm_dict_path) as f:
        csvr = csv.reader(f)
        next(csvr)  # skip header line 1
        next(csvr)  # skip header line 2
        for line in csvr:
            token = line[0]
            dictionary.setdefault(token, len(dictionary))
    dictionary.setdefault(UNK, len(dictionary))

    with gzip.open(data_gzip, 'r') as input:
        data = []
        i = 0
        for _ in tqdm(input):
            _json = ujson.loads(_)
            data.append(_json)
            i += 1
            if i == size: break

    pool = Pool(NUM_PROCESSORS)
    all_contexts = list(tqdm(pool.starmap(make_contexts,
                                          tqdm([(_['content_moses_lemmas'], 5,
                                                 dictionary)
                                                for _ in tqdm(data)]))))
    pool.close()
    del pool

    res = defaultdict(list)
    for c in tqdm(all_contexts):
        sum_contexts(res, c, inplace=True)
    del all_contexts

    selected_items = []
    for k, v in tqdm(res.items()):
        if 10 < len(v) < 600:
            selected_items.append((k, v))
    del res
    random.shuffle(selected_items)

    word_inx_docs = []
    empty_docs_count = 0
    for k, docs in selected_items:
        for doc in docs:
            if len([i for i in doc if i != dictionary[UNK]]) == 0:
                empty_docs_count += 1
            else:
                word_inx_docs.append(doc)

    if dump_flag is True:
        if dump_folder is None or not os.path.exists(dump_folder):
            dump_folder = '/tmp'
        pickle.dump(word_inx_docs, open(os.path.join(dump_folder,
                                                     'word_inx_docs_2.pickle'),
                                        'wb'))
        pickle.dump(dictionary, open(os.path.join(dump_folder,
                                                  'dictionary.pickle'), 'wb'))

    print('Len of word_inx_docs: {}'.format(len(word_inx_docs)))
    print('Empty_docs count: {}'.format(empty_docs_count))
    return word_inx_docs, dictionary


def main(bigartm_dict_path=None,
         data_gzip='../wiki150k',
         size=1000,
         dump_flag=True,
         dump_folder=None):
    if bigartm_dict_path is None:
        print('Run default pipe..')
        return default_pipe(data_gzip=data_gzip,
                            size=size,
                            dump_flag=dump_flag,
                            dump_folder=dump_folder)
    else:
        print('ARTM dict pipe..')
        return bigartm_dict_pipe(
            artm_dict_path=bigartm_dict_path,
            data_gzip=data_gzip,
            size=size,
            dump_flag=dump_flag,
            dump_folder=dump_folder
        )