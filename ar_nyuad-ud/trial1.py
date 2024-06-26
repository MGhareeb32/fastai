import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

from sklearn.cluster import HDBSCAN

def corpus_from_file(filename):
    corpus = pd.read_csv(filename, sep='\t')
    corpus.drop('feats', axis=1, inplace=True)
    corpus.drop('xpos', axis=1, inplace=True)
    return corpus

def corpus_get_setence(corpus, sent_id):
    return corpus.loc[corpus['sent_id'] == sent_id].reset_index()

def corpus_first_n_sentences(corpus, N):
    return corpus.loc[corpus['sent_id'].isin(corpus['sent_id'].unique()[:N])]

def sentence_compare(corpus, sent_id1, sent_id2):
    sent1 = corpus_get_setence(corpus, sent_id1)
    sent2 = corpus_get_setence(corpus, sent_id2)
    dp = {}

    def _compare(i1, i2):
        if (i1, i2) in dp:
            return dp[(i1, i2)]
        dist = float('inf')
        if i1 < len(sent1.index) or i2 < len(sent2.index):
            if i1 < len(sent1.index) and i2 < len(sent2.index):
                t1 = sent1.loc[i1]
                t2 = sent2.loc[i2]
                if t1['upos'] == t2['upos']:
                    dist = min(dist, _compare(i1+1, i2+1))
            if i1 < len(sent1.index):
                dist = min(dist, 1 + _compare(i1+1, i2))
            if i2 < len(sent2.index):
                dist = min(dist, 1 + _compare(i1, i2+1))
        else:
            dist = 1
        dp[(i1, i2)] = dist
        return dist
    return _compare(0, 0)    

def corpus_distance_matrix(corpus):
    sent_ids = corpus['sent_id'].unique()
    N = len(sent_ids)
    dist = np.zeros((N, N))
    for i in tqdm(range(N)):
        for j in tqdm(range(i, N), leave=False):
            dist[i][j] = dist[j][i] = sentence_compare(corpus, sent_ids[i], sent_ids[j])
    return sent_ids, dist

def cluster_corpus(corpus, matrix_fn, min_cluster_size=5):
    print('== Computing distance matrix...')
    sent_ids, matrix = matrix_fn(corpus)
    print('== Clustering...')
    hdb = HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size).fit(matrix)
    print(hdb.labels_)
    print(hdb.probabilities_)

    print('== Aggregating...')
    from collections import defaultdict
    clusters = defaultdict(lambda: [])
    cluster_prob = defaultdict(lambda: 1)
    for prob, sent_id, label in zip(hdb.probabilities_, sent_ids, hdb.labels_):
        clusters[label].append(sent_id)
        cluster_prob[label] *= prob
    print()
    return sorted([(cluster_prob[k], k, v) for k, v in clusters.items()], reverse=True)

def clusters_write_to_file(clusters, corpus, notes, filename):
    print('== Writing to file...')
    with open(filename, 'w') as f:
        f.write(notes)
        f.write("\n========\n")
        for p, label, v in clusters:
            if label >= 0:
                f.write(f"\nCluster {label:02d}:\n")
            else:
                f.write(f"\n\n\nNOT CLUSTERED:\n")
            f.write(f"\t%{p*100:.2f}\n")
            for sent_id in v:
                f.write(' '.join(corpus_get_setence(corpus, sent_id)['form']))
                f.write('\n')

if __name__ == '__main__':
    dev = corpus_from_file('ar_nyuad-ud/short_dev.csv')
    dev_small = corpus_first_n_sentences(dev, 100)
    clusters = cluster_corpus(dev_small, corpus_distance_matrix, 2)
    clusters_write_to_file(clusters, dev_small, """
- linear edit distance
- no regard for keywords, features, etc...
- looks promising, but needs keywords
""", 'ar_nyuad-ud/trial1.txt')
