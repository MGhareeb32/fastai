from transformers import pipeline
import csv
from tqdm import tqdm
import conllu
import random

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

def load_corpus(path):
    with open(path, 'r') as file:
        corpus = conllu.parse(file.read())
    return {sent.metadata['sent_id']: sent for sent in corpus}

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

def sent_text(sent, ids=None, mask_range=None, key='form', same_word_concat=''):
    if ids is None:
        ids = [t['id'] for t in sent]
    if mask_range is None:
        mask_range = (-1, -1)

    ans = []
    watermark = -1
    for t in sent:
        id = t['id']
        if type(id) is int:
            if id not in ids or id <= watermark:
                continue
            ans.append('#' if id >= mask_range[0] and id <= mask_range[1] else t[key])
        else:
            id0, _, id1 = id
            if id0 not in ids and id1 not in ids:
                continue
            conc = []
            for lil_t in sent.filter(id=lambda x: type(x) is int and x >= id0 and x <= id1 and x in ids):
                lil_id = lil_t['id']
                conc.append('#' if lil_id >= mask_range[0] and lil_id <= mask_range[1] else lil_t[key])
                watermark = lil_id
            ans.append(same_word_concat.join(conc))
    return ' '.join(ans).replace('# #', '##').replace('##', '#').replace('#', '[MASK]')

def extract_sentences_of_len(sent, len_range, output):
    def extract(ids):
        if len(ids) < len_range[0] or len(ids) >= len_range[1]:
            return
        res = sent.filter(id=lambda x: x in ids)
        res.metadata['text'] = sent_text(sent, ids)
        output.append((res, sent.metadata['sent_id'], min(ids), max(ids)))
    return extract

if __name__ == '__main__':
    PHRASE_LEN = (2, 3)

    print("\n=== READING CORPUS")
    corpus = load_corpus('ar_nyuad-ud/dev.conllu')
    print(f"{len(corpus):,} full sentences in corpus.")

    print("\n=== PARSING TREE")
    phrases = []
    for sent_id, sent in corpus.items():
        query_subtree(
            sent.to_tree(),
            extract_sentences_of_len(sent, PHRASE_LEN, phrases))
    print(f"Found {len(phrases):,} phrases of length {PHRASE_LEN}.")

    print("\n=== MASKING OUT SENTENCES")
    model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth'
    # model_name = ''CAMeL-Lab/bert-base-arabic-camelbert-ca''
    unmasker = pipeline('fill-mask', model=model_name)
    for phrase, sent_id, mask_low, mask_high in tqdm(phrases[:10]):
        masked_sent = sent_text(corpus[sent_id], mask_range=(mask_low, mask_high))
        print(masked_sent.replace('[MASK]', '؟،،،،،؟'))
        print(f"{sent_text(phrase)} -> ", newline=False)
        for guess in unmasker(masked_sent):
            print(f"  [\'{guess['token_str']}\' {guess['score'] * 100:.1f}%]", newline=False)
        print('')
        print(sent_text(corpus[sent_id], ids=range(mask_low, mask_high+1)))
        print(sent_text(corpus[sent_id], ids=range(mask_low, mask_high+1), key='upos', same_word_concat='+'))
        print('')
