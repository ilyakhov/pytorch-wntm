import pickle
import numpy as np
import os
import random
import argparse
import logging
import pickle

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from input_fn import WNTMDataSet
from model_fn import WNTM_pLSA
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

    if not params.debug:
        dataset_size = params.dataset_size
        try:
            logging.info('Loading data sample...')
            docs = pickle.load(open(
                f'./{dataset_size}_docs_seed{seed}_2.pickle', 'rb'))
        except FileNotFoundError:
            logging.info('Loading whole data. Getting random sample...')
            docs = pickle.load(open(
                os.path.join(params.datapath, 'word_inx_docs_2.pickle'), 'rb'))
            random.shuffle(docs)
            docs = docs[:dataset_size]
            pickle.dump(docs, open(
                f'./{dataset_size}_docs_seed{seed}_2.pickle', 'wb'))
    else:
        logging.info('Loading debug sample...')
        dataset_size = 'sample'
        docs = pickle.load(open('./data_sample/word_inx_docs.pickle', 'rb'))

    logging.info('Loading dictionary...')
    vocab = pickle.load(open(os.path.join(params.datapath,
                                          'dictionary.pickle'), 'rb'))

    if params.__dict__.get('phi_reg_use_vocab_stat', None):
        logging.info('Loading vocab_stat...')
        _vocab_stat = pickle.load(open(os.path.join(params.datapath,
                                                    'vocab_stat.pickle'), 'rb'))
        # crook
        inversed_vocab = {i: v for v, i in vocab.items()}
        vocab_stat = []
        for k in sorted(inversed_vocab.keys()):
            vocab_stat.append(_vocab_stat[inversed_vocab[k]])
    else:
        vocab_stat = None

    context_size = params.context_size
    device = torch.device(params.device)
    dtype = torch.float32
    batch_size = params.batch_size
    num_workers = params.num_workers

    phi_smooth_sparse_tau = params.phi_smooth_sparse_tau
    theta_smooth_sparse_tau = params.theta_smooth_sparse_tau

    dataset = WNTMDataSet(docs, context_size, vocab,
                          dtype, device=None)
    batch_steps = dataset.__len__() // batch_size
    iterator = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)

    train_mode = params.train_mode

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
    p = f'v4_{dataset_size}_ntopics{n_topics}_phi_' \
        f'seed{seed}_'\
        f'reg-phi{phi_smooth_sparse_tau}'\
        f'_reg-thet{theta_smooth_sparse_tau}' \
        f'_epochs{epochs}' \
        f'_m{train_mode}' \
        f'_2steps{two_steps}_num_documents_passes{num_documents_passes}'
    p = os.path.join(dump_phi_folder, p)

    log_perplexity = params.__dict__.get('log_perplexity', False)
    log_matrix_norms = params.__dict__.get('log_matrix_norms', False)

    if two_steps is False:
        logging.info('One-step training...')
        model = WNTM_pLSA(n_topics=n_topics,
                          vocab_size=len(vocab),
                          doc_count=len(docs),
                          batch_size=batch_size,
                          batch_steps=batch_steps,
                          num_collection_passes=epochs,
                          num_documents_passes=num_documents_passes,
                          device=device,
                          dtype=dtype,
                          phi_smooth_sparse_tau=phi_smooth_sparse_tau,
                          theta_smooth_sparse_tau=theta_smooth_sparse_tau,
                          vocab_stat=vocab_stat,
                          mode=train_mode,
                          dump_phi_freq=params.dump_phi_freq,
                          dump_phi_path=p,
                          log_perplexity=log_perplexity,
                          log_matrix_norms=log_matrix_norms
                          )

        model.run(iterator)
        logging.info(model.phi_log)
    else:
        logging.info('Two-steps training...')
        model = WNTM_pLSA(n_topics=n_topics,
                          vocab_size=len(vocab),
                          doc_count=len(docs),
                          batch_size=batch_size,
                          batch_steps=batch_steps,
                          num_collection_passes=epochs1,
                          num_documents_passes=1,
                          device=device,
                          dtype=dtype,
                          phi_smooth_sparse_tau=.0,
                          theta_smooth_sparse_tau=.0,
                          vocab_stat=vocab_stat,
                          mode=train_mode,
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