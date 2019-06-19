import torch
import numpy as np
import pandas as pd


def get_desc(phi, n_objcts, inversed_dictionary=None, th=0.5):
    topic_desc = {}

    collection = pd.DataFrame(phi.T)
    n_topics = collection.shape[0]
    collection.rename({i: f'topic_{n}' for i, n in enumerate(range(n_topics))},
                      axis=0, inplace=True)
    # print(collection[:2])

    non_active_topics = collection.index[collection.sum(axis=1) < th]
    #     logger.warn('Next topics are not active for "%s" modality: %s', '__score_name__', list(non_active_topics))
    collection = collection[~collection.index.isin(non_active_topics)]

    collection.iloc[:, :] = (
            collection.values / collection.sum(axis=1).values[:, np.newaxis])
    # print(collection[:2])
    for topic in collection.index:
        topic_sample = collection.loc[topic].sort_values(ascending=False)[
                       :n_objcts]

        if inversed_dictionary is not None:
            get_word = lambda w: inversed_dictionary.get(w)
        else:
            get_word = lambda w: w

        topic_desc[topic] = ' | '.join(
            ['%s:%3.3f' % (get_word(word), weight)
             for word, weight in topic_sample.iteritems()]
        )

    return topic_desc


if __name__ == '__main__':
    import pickle
    # phi = np.load('../1500000_phi.npy')
    # phi = np.load('../500000_phi_reg-phi-0.001_reg-thet-0.001.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.001_reg-thet-0.01.npy')
    # phi = np.load('../1500000_phi_reg-phi-1e-06_reg-thet-1e-05.npy')
    # phi = np.load('../1500000_phi_reg-phiNone_reg-thetNone.npy')
    # phi = np.load('../1500000_phi_reg-phiNone_reg-thetNone_epochs15.npy')
    # phi = np.load('../1500000_phi_reg-phi-1e-06_reg-thet-1e-05_epochs15.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.001_reg-thet-0.001_epochs15.npy')
    # phi = np.load('../1500000_phi_reg-phi-1e-06_reg-thet-1e-06_epochs15.npy')
    # phi = np.load('../1500000_phi_reg-phi-1e-06_reg-thet-1e-06_epochs5.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.001_reg-thet-0.001_epochs10.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.001_reg-thet-0.001_epochs10.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.01_reg-thet-0.01_epochs10.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.01_reg-thet-0.01_epochs5.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.001_reg-thet-0.001_epochs5_mv1.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.01_reg-thet-0.01_epochs5_mv1.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.1_reg-thet-1.0_epochs5_mv1.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.1_reg-thet-1.0_epochs10_mv1.npy')
    # phi = np.load('../1500000_phi_reg-phi-0.1_reg-thet-1.0_epochs7_mv1.npy')
    # phi = np.load('../1500000_phi_reg-phiNone_reg-thetNone_epochs5_mv1.npy')
    phi = np.load('../1500000_phi_reg-phiNone_reg-thetNone_epochs25_mv1.npy')
    vocab = pickle.load(open('../data_sample/dictionary.pickle', 'rb'))
    inversed_vocab = {i: v for v, i in vocab.items()}

    for t, words in get_desc(phi, 15, inversed_vocab, th=0.0).items():
        print(f'{t} — {words}\n')

