import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from zss import simple_distance, Node

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

from sklearn.cluster import HDBSCAN
from tqdm import tqdm

def corpus_from_file(filename):
    corpus = pd.read_csv(filename, sep='\t')
    corpus.drop('feats', axis=1, inplace=True)
    corpus.drop('xpos', axis=1, inplace=True)
    return corpus

def print_tree(node, depth=0):
    print(f"{'|   ' * depth}{node.label}")
    for ch in node.children:
        print_tree(ch, depth+1)

def corpus_get_sentence(corpus, sent_id):
    return corpus.loc[corpus['sent_id'] == sent_id].reset_index()
        
def sentence_to_tree(sent):
    def _sentence_to_tree(index):
        token = sent.loc[index]
        graph_node = Node(token['upos'])
        # print(sent_node)
        # sent_node_id = sent_node['token_id'].item()
        # print(sent_node_id)
        for ch in sent.index[sent['head_id'] == token['token_id']].tolist():
            graph_node.addkid(_sentence_to_tree(ch))
        return graph_node

    return _sentence_to_tree(sent.index[sent['head_dist'] == 0][0])

def sentence_distance(sent1, sent2):
    return simple_distance(
        sentence_to_tree(sent1),
        sentence_to_tree(sent2)
    )

def corpus_first_n_sentences(corpus, N):
    return corpus.loc[corpus['sent_id'].isin(corpus['sent_id'].unique()[:N])]

def corpus_distance_matrix(corpus):
    sent_ids = corpus['sent_id'].unique()
    N = len(sent_ids)
    dist = np.zeros((N, N))
    for i in tqdm(range(N)):
        for j in tqdm(range(i, N), leave=False):
            dist[i][j] = dist[j][i] = sentence_distance(
                corpus_get_sentence(corpus, sent_ids[i]),
                corpus_get_sentence(corpus, sent_ids[j]))
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
                f.write(' '.join(corpus_get_sentence(corpus, sent_id)['form']))
                f.write('\n')


if __name__ == '__main__':
    dev = corpus_from_file('ar_nyuad-ud/short_dev.csv')
    dev_small = corpus_first_n_sentences(dev, 100)
    clusters = cluster_corpus(dev_small, corpus_distance_matrix, 2)
    clusters_write_to_file(clusters, dev_small, """
- tree edit distance
- no regard for keywords, features, etc...
- cant tell if this is better or linear
""", 'ar_nyuad-ud/trial2.txt')
