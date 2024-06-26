import torch
from transformers import pipeline
import csv
from tqdm import tqdm
import conllu
from collections import defaultdict
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from sklearn.cluster import HDBSCAN
from difflib import SequenceMatcher

# LOGS

log_file = None
def print(*text, newline=True, escape_newlines=False):
    global log_file
    if log_file is None:
        log_file = open(__file__.replace('.py', '.log'), 'w', encoding='utf-8')
    for t in text:
        txt = str(t)
        if escape_newlines: txt = txt.replace('\n', '\\n')
        log_file.write(txt)

        if newline: log_file.write('\n')
    log_file.flush()

# READING CORPUS

def load_corpus(paths):
    ans = {}
    for path in paths:
        with open(path, 'r') as file:
            corpus = conllu.parse(file.read())
        ans.update({sent.metadata['sent_id']: sent for sent in corpus})
    return ans

# PARSE TREE

def query_subtree(node, callback_fn=None, depth=0, parent=None):
    total_size = 1
    total_ids = set([node.token['id']])
    for ch in node.children:
        size, ids = query_subtree(ch, callback_fn, depth+1, node)
        total_size += size
        total_ids = total_ids.union(ids)
    is_conseq = total_size == len(total_ids) and total_size == (max(total_ids) - min(total_ids) + 1)
    if is_conseq and callback_fn is not None:
        callback_fn(total_ids)
    return total_size, total_ids

def sent_text(sent, highlight_id=-1):
    ans = []
    for t in sent:
        if type(t['id']) is int:
            if t['id'] == highlight_id:
                ans.append(f"***{t['form']}***")
            else:
                ans.append(t['form'])
    return ' '.join(ans)

def extract_sentences_of_len(sent, len_range, output):
    def extract(ids):
        if len(ids) < len_range[0] or len(ids) >= len_range[1]:
            return
        res = sent.filter(id=lambda x: x in ids)
        res.metadata['text'] = sent_text(sent, ids)
        output.append((res, sent.metadata['sent_id'], min(ids), max(ids)))
    return extract

# STATS

def token_synth_key(t, sent):
    upos = t['upos']
    deprel = t['deprel']
    head_id = t['head']
    head = sent.filter(id=head_id)
    head_upos = head[0]['upos'] if len(head) > 0 else '#'
    return f"{upos}>{deprel}>{head_upos}"

def collect_stats(corpus):
    stats = defaultdict(lambda: [0, defaultdict(lambda: 0)])
    for sent_id, sent in tqdm(corpus.items()):
        for t in sent:
            if type(t['id']) is not int: continue
            stat = stats[token_synth_key(t, sent)]
            stat[0] += 1
            stat[1][t['form']] += 1
    # sort forms for each key
    for k in stats.keys():
        stats[k][1] = [
            f for c, f in
            sorted(
                [
                    (count, form) for form, count in stats[k][1].items()
                ],
                reverse=True)
            if c > stats[k][0] / 1000
        ]
    # sort overall stats
    sorted_stats = sorted([(v, k) for k, v in stats.items()], reverse=True)
    return { k: v for v, k in sorted_stats }

# SYNTH

def synth_rules_load():
    ans = {}
    with open(__file__.replace('.py', '.csv'), 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for count, link, repl in reader:
            ans[link] = repl
    return ans
word2synth = synth_rules_load()

def token_synth(t, sent):
    def simi(a, b): return SequenceMatcher(None, a, b).ratio()
    key = token_synth_key(t, sent)
    
    keys_ranked = sorted([
        (simi(key.split('>')[0], dict_key.split('>')[0]) + simi(key, dict_key) / 10, dict_key)
        for dict_key in word2synth.keys()
    ], reverse=True)
    value = word2synth[keys_ranked[0][1]]

    if value == '"=':
        return t['form']
    if value == '"x':
        return ''
    return value
    
def sent_synth(sent):            
    ans = []
    for i, t in enumerate(sent):
        if type(t['id']) is int:
            ans.append(token_synth(t, sent))
    return ' '.join(ans).replace('  ', ' ')

# EMBED

def print_similarity(idx_range, phrases, embeddings):
    txts = [sent_text(s[0]) for s in phrases[idx_range[0]:idx_range[1]]]
    matrix = model.similarity(embeddings[idx_range[0]:idx_range[1],:], embeddings[idx_range[0]:idx_range[1],:])
    for i1 in range(len(matrix)):
        for i2 in range(len(matrix)):
            print(f"  {matrix[i1][i2]:.3f}", newline=False)
        print(f"  =  {txts[i1]}")
    
# MAIN

if __name__ == '__main__':
    PHRASE_LEN = (5, 6)
    print("\n=== READING CORPUS")
    # corpus = load_corpus(['ar_nyuad-ud/train.conllu', 'ar_nyuad-ud/dev.conllu', 'ar_nyuad-ud/test.conllu'])
    corpus = load_corpus(['ar_nyuad-ud/dev.conllu'])
    print(f"{len(corpus):,} full sentences in corpus.")

    print("\n=== PARSING TREE")
    phrases = []
    for sent_id, sent in corpus.items():
        query_subtree(
            sent.to_tree(),
            extract_sentences_of_len(sent, PHRASE_LEN, phrases))
    print(f"Found {len(phrases):,} phrases of length {PHRASE_LEN}.")

    # print("\n=== STATS")
    # stats_csv_writer = csv.writer(log_file, quoting=csv.QUOTE_NONNUMERIC)
    # for stat, val in collect_stats(corpus).items():
    #     count, examples = val
    #     stats_csv_writer.writerow([count, stat, ' '.join(examples)])

    print("\n=== SYNTH")
    synth_sentences = list(map(sent_synth, [p[0] for p in phrases]))
    for i, sent in enumerate(phrases):
        if i == 6: break
        print(sent_text(sent[0]))
        print(synth_sentences[i])
        print('')

    print("\n=== EMBED")
    model_name = "sentence-transformers/paraphrase-distilroberta-base-v1"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(synth_sentences)
    print_similarity((0, 10), phrases, embeddings)

    print("\n=== CLUSTERING")
    hdb = HDBSCAN()
    hdb.fit(embeddings)
    clusters = defaultdict(lambda: [])
    for sent, prob, label in zip(phrases, hdb.probabilities_, hdb.labels_):
        clusters[label].append((sent, prob))
    print(f"{len(clusters)-1:,} clusters found.")
    print(f"Failed to cluster {len(clusters[-1]):,}/{len(phrases):,} phrases.")

    with open(__file__.replace('.py', '.cluster.csv'), 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for cluster_id, sent_probs in tqdm(sorted(clusters.items(), reverse=True)):
            for sent, prob in sent_probs:
                small_sent, source_sent_id, min_id, max_id = sent
                writer.writerow([
                    cluster_id, (prob * 10000 // 100) / 100, sent_text(small_sent), sent_synth(small_sent),
                    source_sent_id, min_id, max_id
                ])
            csvfile.flush()