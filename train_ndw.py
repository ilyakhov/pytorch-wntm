import pickle
import numpy as np
import os
import random
import argparse
import logging
import pickle
from copy import deepcopy

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from input_fn import n_dwDataSet
from model_fn import pLSA
from utils.utils import *
from utils.get_desc import get_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='configs/config.json')
    FLAGS = parser.parse_args()
    params = Params(jsonpath=FLAGS.config)
    set_logger(params.log_path)
    logging.info(params.dumps())

    seed = params.seed

    if seed:
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if params.sklearn is True:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer

        n_dw_datapath = './n_dw'

        try:
            n_dw, vocab, inversed_vocab = pickle.load(
                open(os.path.join(n_dw_datapath,
                                  f'20news_{params.max_feats}vocab.pkl'), 'rb'))
        except FileNotFoundError:
            logging.info('Fetching 20newsgroups...')
            data = fetch_20newsgroups(data_home='/tmp/20news', subset='train',
                                      categories=None, shuffle=True,
                                      random_state=42, remove=(),
                                      download_if_missing=True).data

            # params.max_feats = 1000
            #
            cv = CountVectorizer(max_features=params.max_feats,
                                 stop_words='english')
            n_dw = np.array(cv.fit_transform(data).todense())
            logging.info(f'n_dw.shape: {n_dw.shape}')

            vocab = cv.vocabulary_
            inversed_vocab = {i: v for v, i in vocab.items()}

            n_dw_nonempty = []
            for i in deepcopy(np.arange(n_dw.shape[0])):
                if np.sum(n_dw[i]) > 0:
                    n_dw_nonempty.append(n_dw[i])

            n_dw = np.array(n_dw_nonempty)
            del n_dw_nonempty

            logging.info(f'n_wd.shape: {n_dw.shape}')
            if not os.path.exists(n_dw_datapath):
                make_directory(n_dw_datapath)
            pickle.dump([n_dw, vocab, inversed_vocab],
                open(os.path.join(n_dw_datapath,
                                  f'20news_{params.max_feats}vocab.pkl'), 'wb'))
    else:
        raise NotImplementedError

    device = torch.device(params.device)
    dtype = torch.float32
    batch_size = params.batch_size
    num_workers = params.num_workers

    phi_smooth_sparse_tau = params.phi_smooth_sparse_tau
    theta_smooth_sparse_tau = params.theta_smooth_sparse_tau

    dataset = n_dwDataSet(n_dw, dtype, device=None)
    batch_steps = dataset.__len__() // batch_size
    iterator = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)

    two_steps = params.two_steps
    if two_steps is False:
        epochs = params.num_collection_passes
    else:
        epochs1 = params.num_collection_passes_step1
        epochs2 = params.num_collection_passes_step2

    n_topics = params.n_topics
    num_documents_passes = params.num_documents_passes

    epochs = epochs1 + epochs2 if two_steps else epochs

    dump_phi_folder = './trained_phi/'
    if not os.path.exists(dump_phi_folder):
        make_directory(dump_phi_folder)
    p = f'ndw_vocab{len(vocab)}_ntopics{n_topics}_phi_' \
        f'reg-phi{phi_smooth_sparse_tau}'\
        f'_reg-thet{theta_smooth_sparse_tau}' \
        f'_epochs{epochs}' \
        f'_2steps{two_steps}_num_documents_passes{num_documents_passes}'
    p = os.path.join(dump_phi_folder, p)

    log_perplexity = params.__dict__.get('log_perplexity', False)
    log_matrix_norms = params.__dict__.get('log_matrix_norms', False)

    if two_steps is False:
        logging.info('One-step training...')
        model = pLSA(n_topics=n_topics,
                     vocab_size=len(vocab),
                     doc_count=n_dw.shape[0],
                     batch_size=batch_size,
                     batch_steps=batch_steps,
                     num_collection_passes=epochs,
                     num_documents_passes=num_documents_passes,
                     device=device,
                     dtype=dtype,
                     phi_smooth_sparse_tau=phi_smooth_sparse_tau,
                     theta_smooth_sparse_tau=theta_smooth_sparse_tau,
                     dump_phi_freq=params.dump_phi_freq,
                     dump_phi_path=p,
                     log_perplexity=log_perplexity,
                     log_matrix_norms=log_matrix_norms
                     )

        model.run(iterator)
        logging.info(model.phi_log)
    else:
        logging.info('Two-steps training...')
        model = pLSA(n_topics=n_topics,
                     vocab_size=len(vocab),
                     doc_count=n_dw.shape[0],
                     batch_size=batch_size,
                     batch_steps=batch_steps,
                     num_collection_passes=epochs,
                     num_documents_passes=num_documents_passes,
                     device=device,
                     dtype=dtype,
                     phi_smooth_sparse_tau=.0,
                     theta_smooth_sparse_tau=.0,
                     dump_phi_freq=params.dump_phi_freq,
                     dump_phi_path=p,
                     log_perplexity=log_perplexity,
                     log_matrix_norms=log_matrix_norms
                     )
        model.run(iterator)
        model.num_collection_passes = epochs2
        model.phi_smooth_sparse_tau = phi_smooth_sparse_tau
        model.theta_smooth_sparse_tau = theta_smooth_sparse_tau
        model.run(iterator)
        logging.info(model.phi_log)

    phi = model.get_phi()
    logging.info('Phi dump path: {p}.npy'.format(p=p))

    if params.dump_phi is True:
        np.save(p, phi)

    if params.desc2log is True:
        inversed_vocab = {i: v for v, i in vocab.items()}
        for t, words in get_desc(phi, 25, inversed_vocab, th=0.5).items():
            logging.info(f'{t} — {words}')