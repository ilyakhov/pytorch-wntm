import torch
import numpy as np
import pandas as pd


def get_desc(phi, n_objcts, inversed_dictionary=None, th=0.5):
    topic_desc = {}

    collection = pd.DataFrame(phi.T)
    n_topics = collection.shape[0]
    collection.rename({i: f'topic_{n}' for i, n in enumerate(range(n_topics))},
                      axis=0, inplace=True)

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
    phi = np.load('../1500000_phi_reg-phiNone_reg-thetNone_epochs25_mv1.npy')
    vocab = pickle.load(open('../data_sample/dictionary.pickle', 'rb'))
    inversed_vocab = {i: v for v, i in vocab.items()}

    for t, words in get_desc(phi, 15, inversed_vocab, th=0.0).items():
        print(f'{t} — {words}\n')

