import conllu
from collections import defaultdict
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from sklearn.cluster import HDBSCAN
from collections import defaultdict
import csv
from tqdm import tqdm

# READING CORPUS

def load_corpus(path):
    with open(path, 'r') as file:
        corpus = conllu.parse(file.read())
    return {sent.metadata['sent_id']: sent for sent in corpus}

log_file = None
def print(*text, newline=True, escape_newlines=False):
    global log_file
    if log_file is None:
        log_file = open('ar_nyuad-ud/trial6.log', 'w', encoding='utf-8')
    for t in text:
        txt = str(t)
        if escape_newlines: txt = txt.replace('\n', '\\n')
        log_file.write(txt)

        if newline: log_file.write('\n')
    log_file.flush()

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

def sent_to_pos(sent, mask_range=None):
    ans = []
    masking_started = False
    for t in sent:
        if type(t['id']) is int:
            txt = None
            if mask_range is not None and t['id'] >= mask_range[0] and t['id'] <= mask_range[1]:
                if not masking_started:
                    txt = '<mask>'
                    masking_started = True
            else:
                txt = t['upos']
            if txt is not None: ans.append(txt)
    return ' '.join(ans)

# EMBEDDINGS

def print_similarity(idx_range, phrases):
    txts = [s[0].metadata['text'] for s in phrases[idx_range[0]:idx_range[1]]]
    matrix = model.similarity(embeddings[idx_range[0]:idx_range[1],:], embeddings[idx_range[0]:idx_range[1],:])
    for i1 in range(len(matrix)):
        for i2 in range(len(matrix)):
            print(f"  {matrix[i1][i2]:.3f}", newline=False)
        print(f"  =  {txts[i1]}")

def model_sentences(data):
    model_name = "bert-base-uncased"
    # model_name = "ProsusAI/finbert"
    # model_name = "BAAI/bge-small-en-v1.5"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataset = datasets.DenoisingAutoEncoderDataset(data)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=model_name, tie_encoder_decoder=True
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        weight_decay=0.1,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
    )
    return model

if __name__ == '__main__':
    PHRASE_LEN = (5, 6)
    print("=== READING CORPUS")
    corpus = load_corpus('ar_nyuad-ud/dev.conllu')
    print(f"{len(corpus):,} full sentences in corpus.")

    print("=== PARSING TREE")
    phrases = []
    for sent_id, sent in corpus.items():
        query_subtree(
            sent.to_tree(),
            extract_sentences_of_len(sent, PHRASE_LEN, phrases))
    print(f"Found {len(phrases):,} phrases of length {PHRASE_LEN}.")

    print("=== EMBEDDING SENTENCES")
    model = model_sentences([sent_to_pos(s[0]) for s in phrases])
    embeddings = model.encode(list(map(sent_to_pos, [s[0] for s in phrases])))
    print_similarity((0, 10), phrases)

    print("=== CLUSTERING")
    hdb = HDBSCAN()
    hdb.fit(embeddings)
    clusters = defaultdict(lambda: [])
    for sent, prob, label in zip(phrases, hdb.probabilities_, hdb.labels_):
        clusters[label].append((sent, prob))
    print(f"{len(clusters)-1:,} clusters found.")
    print(f"Failed to cluster {len(clusters[-1]):,}/{len(phrases):,} phrases.")

    with open('ar_nyuad-ud/trial6.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for cluster_id, sents in tqdm(sorted(clusters.items(), reverse=True)):
            for sent_prob in sents:
                sent, prob = sent_prob
                small_sent, source_sent_id, min_id, max_id = sent
                writer.writerow([
                    cluster_id, prob,
                    small_sent.metadata['text'],
                    sent_to_pos(small_sent),
                    source_sent_id, min_id, max_id,
                    sent_to_pos(corpus[source_sent_id], (min_id, max_id)),
                ])
            csvfile.flush()
