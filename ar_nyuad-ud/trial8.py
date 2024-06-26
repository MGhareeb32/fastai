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

def sent_text(sent, ids):
    ans = []
    watermark = -1
    for t in sent:
        id = t['id']
        if type(id) is int:
            if id not in ids or id <= watermark:
                continue
            ans.append(t['form'])
        else:
            id0, _, id1 = id
            if id0 not in ids and id1 not in ids:
                continue
            conc = []
            for lil_t in sent.filter(id=lambda x: type(x) is int and x >= id0 and x <= id1 and x in ids):
                conc.append(lil_t['form'])
                watermark = lil_t['id']
            ans.append(''.join(conc))
    return ' '.join(ans)

def extract_sentences_of_len(sent, len_range, output):
    def extract(ids):
        if len(ids) < len_range[0] or len(ids) >= len_range[1]:
            return
        res = sent.filter(id=lambda x: x in ids)
        res.metadata['text'] = sent_text(sent, ids)
        output.append((res, sent.metadata['sent_id'], min(ids), max(ids)))
    return extract

# SYNTHESIZE

POS2LONG = {
    'PR': 'PRON',
    'NN': 'NOUN',
    'PN': 'PROPN',
    'AJ': 'ADJ',
    'AV': 'ADV',

    'VB': 'VERB',
    'AX': 'AUX',

    'DT': 'DET',
    'AP': 'ADP',
    'CJ': 'CCONJ',
    'SJ': 'SCONJ',
    'PT': 'PART',

    'NM': 'NUM',
    'IJ': 'INTJ',
    'P.': 'PUNCT',
    'XX': 'X',
}
LONG2POS = { v: k for k, v in POS2LONG.items() }
SHORT2FEAT = {}

def synth_token(t):
    upos = LONG2POS[t['upos']]

    feats = dict(t['feats']) if t['feats'] is not None else {}
    letters = ''
    [1, .5, 1, 1, .5, .5, 1, 1, 1, 1, 1, .5]
    for f in ['Definite', 'Case', 'Gender', 'Number', 'Aspect', 'Mood', 'Person', 'Voice', 'AdpType', 'PronType', 'Polarity', 'NumForm']:
        if f in feats:
            key = f"{f}-{feats[f][0]}"
            if key in SHORT2FEAT and SHORT2FEAT[key] != feats[f]:
                print(f"{key} common between {feats[f]} and {SHORT2FEAT[key]}")
                exit()
            SHORT2FEAT[key] = feats[f]
            letters += feats[f][0]
            del feats[f]
        else:
            letters += '?'
    return f"{upos}-{letters}"

def sent_text(sent, highlight_id=-1):
    ans = []
    for t in sent:
        if type(t['id']) is int:
            if t['id'] == highlight_id:
                ans.append(f"***{t['form']}***")
            else:
                ans.append(t['form'])
    return ' '.join(ans)

def synth(t, stats):
    return stats[synth_token(t)][0][1]

# SYNTH

def synth_csv_load():
    synth2word = defaultdict(lambda: {})
    with open(__file__.replace('.py', '.csv'), 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for pos, feats, *value in reader:
            synth2word[pos][feats] = value
    return {k: v for k, v in synth2word.items()}
SYNTH2WORD = synth_csv_load()

def sent_synth(sent):
    def _closest(f1, mapping):
        f = None
        best_dist = len(f1)
        for f2 in mapping.keys():
            dist = 0
            for i in range(len(f1)):
                if f1[i] != f2[i]: dist += 1
            if dist < best_dist:
                best_dist = dist
                f = f2
        return mapping[f]
            
    ans = []
    for i, t in enumerate(sent):
        if type(t['id']) is int:
            pos, feats = synth_token(t).split('-')
            if pos in SYNTH2WORD:
                options = _closest(feats, SYNTH2WORD[pos])
                ans.append(options[i % len(options)])
            else:
                ans.append(t['form'])
    return ' '.join(ans)

# EMBED

def print_similarity(idx_range, phrases, embeddings):
    txts = [sent_text(s[0]) for s in phrases[idx_range[0]:idx_range[1]]]
    matrix = model.similarity(embeddings[idx_range[0]:idx_range[1],:], embeddings[idx_range[0]:idx_range[1],:])
    for i1 in range(len(matrix)):
        for i2 in range(len(matrix)):
            print(f"  {matrix[i1][i2]:.3f}", newline=False)
        print(f"  =  {txts[i1]}")

# TRANSLATE

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

    print("\n=== SYNTH")
    # https://docs.google.com/spreadsheets/d/14-Cvypa-YyGacMNU_uabPhW8ZC02Js1BmzcbgZX_CW8/
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

    with open(__file__.replace('.py', '.cluster'), 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for cluster_id, sent_probs in tqdm(sorted(clusters.items(), reverse=True)):
            for sent, prob in sent_probs:
                small_sent, source_sent_id, min_id, max_id = sent
                writer.writerow([
                    cluster_id, (prob * 10000 // 100) / 100, sent_text(small_sent), sent_synth(small_sent),
                    source_sent_id, min_id, max_id
                ])
            csvfile.flush()

