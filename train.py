import pickle
import numpy as np
import os
import random

from input_fn import WNTMDataSet
from model_fn import WNTM_pLSA

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # debug sample
    # docs = pickle.load(open('./data_sample/word_inx_docs.pickle', 'rb'))
    # vocab = pickle.load(open('./data_sample/dictionary.pickle', 'rb'))

    # dataset_size = 3000000  # with tf.float16 overflowing: nan
    # # dataset_size = 500000
    seed = 4242
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    dataset_size = 2000000
    # dataset_size = 1400000
    # # #
    try:
        docs = pickle.load(open(f'./{dataset_size}_docs_seed{seed}.pickle', 'rb'))
    except FileNotFoundError:
        docs = pickle.load(open('./data/word_inx_docs.pickle', 'rb'))
        random.shuffle(docs)
        docs = docs[:dataset_size]
        pickle.dump(docs, open(f'./{dataset_size}_docs_seed{seed}.pickle', 'wb'))

    # dataset_size = 'sample'
    # docs = pickle.load(open('./data_sample/word_inx_docs.pickle', 'rb'))

    vocab = pickle.load(open('./data/dictionary.pickle', 'rb'))
    _vocab_stat = pickle.load(open('./data/vocab_stat.pickle', 'rb'))
    # crook
    inversed_vocab = {i: v for v, i in vocab.items()}
    vocab_stat = []
    for k in sorted(inversed_vocab.keys()):
        vocab_stat.append(_vocab_stat[inversed_vocab[k]])

    # print(docs[0])
    context_size = 5
    device = torch.device('cuda:0')
    dtype = torch.float32
    batch_size = 10000
    num_workers = 12
    # phi_smooth_sparse_tau = None
    # theta_smooth_sparse_tau = None

    phi_smooth_sparse_tau = -1e+5  # -1e-2  #-1e-2
    theta_smooth_sparse_tau = -1e+5    #-1e-2  #-1e-2

    # phi_smooth_sparse_tau = -.1  # -1e-2  #-1e-2
    # theta_smooth_sparse_tau = -.01    #-1e-2  #-1e-2

    dataset = WNTMDataSet(docs, context_size, vocab,
                          dtype, device=None)
    batch_steps = dataset.__len__() // batch_size
    iterator = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)

    # train_mode = 'v2'
    train_mode = 'v1'
    # epochs = 25
    # epochs = 100
    epochs = 30

    epochs1 = 20
    epochs2 = 10

    n_topics = 400
    two_steps = True
    num_documents_passes = 1

    epochs = epochs1 + epochs2 if two_steps else epochs
    p = f'./v2_{dataset_size}_ntopics{n_topics}_phi_' \
        f'reg-phi{phi_smooth_sparse_tau}'\
        f'_reg-thet{theta_smooth_sparse_tau}' \
        f'_epochs{epochs}' \
        f'_m{train_mode}' \
        f'_2steps{two_steps}_num_documents_passes{num_documents_passes}'

    if two_steps is False:
        print('One-step training...')
        model = WNTM_pLSA(n_topics=n_topics,
                          vocab_size=len(vocab),
                          doc_count=len(docs),
                          context_size=context_size,
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
                          dump_phi_freq=5,
                          dump_phi_path=p
                          )

        model.run(iterator)
        print(model.phi_log)
    else:
        print('Two-steps training...')
        model = WNTM_pLSA(n_topics=n_topics,
                          vocab_size=len(vocab),
                          doc_count=len(docs),
                          context_size=context_size,
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
                          dump_phi_freq=5,
                          dump_phi_path=p
                          )
        model.run(iterator)
        print(model.phi_log)

        model.num_collection_passes = epochs2
        model.phi_smooth_sparse_tau = phi_smooth_sparse_tau
        model.theta_smooth_sparse_tau = theta_smooth_sparse_tau
        print(model.phi_smooth_sparse_tau, model.theta_smooth_sparse_tau)
        model.run(iterator)
        print(model.phi_log)

    phi = model.get_phi()
    if os.path.exists(p):
        try:
            i = int(p.rsplit('-')[-1])
            p = p.rsplit('-')[0] + f'-{i+1}'
        except:
            p += '-' + str(0)
    print(p + '.npy')
    np.save(p, phi)