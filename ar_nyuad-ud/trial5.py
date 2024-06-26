import conllu
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import TrainingArguments, Trainer
import torch
import math
from transformers import pipeline
import csv
from tqdm import tqdm
from collections import defaultdict

# READING CORPUS

def load_corpus(path):
    with open(path, 'r') as file:
        corpus = conllu.parse(file.read())
    return {sent.metadata['sent_id']: sent for sent in corpus}

log_file = None
def print(*text, newline=True, escape_newlines=False):
    global log_file
    if log_file is None:
        log_file = open('ar_nyuad-ud/trial5.log', 'w', encoding='utf-8')
    for t in text:
        txt = str(t)
        if escape_newlines: txt = txt.replace('\n', '\\n')
        log_file.write(txt)

        if newline: log_file.write('\n')
    log_file.flush()

def sent_tree_traverse(node, callback_fn=None, depth=0, parent=None):
    total_size = 1
    subtree_ids = set([node.token['id']])
    for ch in node.children:
        size, ids = sent_tree_traverse(ch, callback_fn, depth+1, node)
        total_size += size
        subtree_ids = subtree_ids.union(ids)

    is_conseq = total_size == len(subtree_ids) and total_size == (max(subtree_ids) - min(subtree_ids) + 1)
    if is_conseq and callback_fn is not None:
        callback_fn(node, parent, depth, subtree_ids)

    return total_size, subtree_ids

def sent_key(sent, key='upos', r=False):
    reverse_fn = reversed if r else list
    return [t[key] for t in reverse_fn(sent) if type(t['id']) is int]

# PRE-PROCESSING

TOKENIZATION_BLOCK_SIZE = 32
TRAINING_BATCH_SIZE = 32 * 5
POS2WORD = {
    'INTJ': ':',
    'PUNCT': '\n',
    'CCONJ': ';',
    'PART': '?',
    'SCONJ': 'if',
    'DET': 'type',
    'NUM': '3',
    'ADJ': '*',
    'NOUN': '_',
    'VERB': 'var',
    'X': '!',
    'PRON': 'self',
    'AUX': '=',
    'ADV': 'raise',
    'ADP': 'in',
    'PROPN': 'def',

    # 'ADP': 'A',
    # 'ADV': 'B',
    # 'CCONJ': 'C',
    # 'DET': 'D',
    # 'PROPN': 'E',
    # 'INTJ': 'I',
    # 'ADJ': 'J',
    # 'NOUN': 'N',
    # 'PRON': 'O',
    # 'PUNCT': 'P',
    # 'X': 'Q',
    # 'NUM': 'R',
    # 'SCONJ': 'S',
    # 'PART': 'T',
    # 'VERB': 'V',
    # 'AUX': 'X',

    # 'INTJ': 'interjection',
    # 'PUNCT': ',',
    # 'CCONJ': 'coordinate',
    # 'PART': 'particle',
    # 'SCONJ': 'subordinate',
    # 'DET': 'determine',
    # 'NUM': 'number',
    # 'ADJ': 'adjective',
    # 'NOUN': 'noun',
    # 'VERB': 'verb',
    # 'X': 'x',
    # 'PRON': 'pronoun',
    # 'AUX': 'auxilary',
    # 'ADV': 'adverb',
    # 'ADP': 'adposition',
    # 'PROPN': 'proper',
}
WORD2POS = {v: k for k, v in POS2WORD.items()}
print(f"WORD2POS={len(WORD2POS)} POS2WORD={len(POS2WORD)}")
assert(len(WORD2POS) == len(POS2WORD))

def sent_process_sample(id_and_sent, conseq_len, sample_count=1):
    def _extract(node, parent, depth, conseq_ids):
        if len(conseq_ids) < conseq_len[0] or len(conseq_ids) >= conseq_len[1]:
            return
        extracted.append((orig_sent, min(conseq_ids), max(conseq_ids)))

    def _processed_sent(sent, min_id, max_id):
        sent_parts = []
        for t in sent:
            t_id = t['id']
            if type(t_id) is int:
                t_processed = POS2WORD[t['upos']]
                if t_id >= min_id and t_id <= max_id:
                    sent_parts.append(f"{t_processed}") # TODO
                else:
                    sent_parts.append(t_processed)
        return(sent_parts)

    sent_id, orig_sent = id_and_sent
    extracted = []
    sent_tree_traverse(orig_sent.to_tree(), _extract)
    samples = random.sample(extracted, sample_count)
    res = []
    for sent, min_id, max_id in samples:
        processed_sent = _processed_sent(sent, min_id, max_id)
        res.append(' '.join(processed_sent))
    return sent_id, res

def my_zip(a, b):
    return [f"{x}x{y}" for x, y in zip(a, b)]

# TOKENIZING

def dataset_tokenize(dataset):
    def tokenize(x):
        result = tokenizer(x["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
        
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    return tokenizer, tokenized_dataset

def dataset_to_labeled_blocks(rows):
    concatenated_row = {k: sum(rows[k], []) for k in rows.keys()}
    total_length = len(concatenated_row['input_ids'])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    assert(total_length >= TOKENIZATION_BLOCK_SIZE)
    blocky_total_length = (total_length // TOKENIZATION_BLOCK_SIZE) * TOKENIZATION_BLOCK_SIZE
    print(f"    Dropping {total_length - blocky_total_length} tokens / {total_length}.")
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + TOKENIZATION_BLOCK_SIZE] for i in range(0, blocky_total_length, TOKENIZATION_BLOCK_SIZE)]
        for k, t in concatenated_row.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

# COLLATING

# FINETUNEING

# PREDICTING

# MAIN

if __name__ == '__main__':
    torch.cuda.empty_cache()
    CAPACITY = 48*4*32

    print(f'\n=== READING CORPUS')
    corpus = load_corpus('ar_nyuad-ud/dev.conllu')
    print(f"corpus.len = {len(corpus)} sentences")

    print(f'\n=== PRE-PROCESSING')
    processed_sents = list(map(lambda sent: sent_process_sample(sent, (1, 2)), corpus.items()))
    assert(len(processed_sents) == len(corpus))
    print(' '.join(sent_key(corpus[processed_sents[2][0]])))
    print(' '.join([POS2WORD[w] for w in sent_key(corpus[processed_sents[2][0]])]), escape_newlines=True)
    print(my_zip(
        processed_sents[2][1][0].split(' '),
        sent_key(corpus[processed_sents[2][0]], 'form'),
    ))

    print(f'\n=== CREATING DATASET')
    dataset = Dataset.from_dict({
        'sent_id': [m[0] for m in processed_sents],
        'text': [m[1][0] for m in processed_sents],
    }).train_test_split(test_size=0.2)
    print(my_zip(
        dataset['train'][2]['text'].split(' '),
        sent_key(corpus[dataset['train'][2]['sent_id']], 'form'),
    ))

    print(f'\n=== TOKENIZING')
    tokenizer, tokenized_dataset = dataset_tokenize(dataset)
    print(f"tokenizer.model_max_length = {tokenizer.model_max_length}")
    print(tokenizer.decode(tokenized_dataset['train'][2]['input_ids']), escape_newlines=True)
    print(tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][2]['input_ids']))
    print(tokenized_dataset['train'][2])
    for pos in POS2WORD.values():
        tokens = tokenizer(pos)
        assert(len(tokens) <= 3)
        print(f"    {pos: <12}{tokens}", escape_newlines=True)

    print(f'\n=== TURNING INTO BLOCKS')
    blocky_dataset = tokenized_dataset.map(dataset_to_labeled_blocks, batched=True, num_proc=4)
    print(blocky_dataset['train'][10])
    print(tokenizer.decode(blocky_dataset["train"][10]['labels']), escape_newlines=True)

    print(f'\n=== COLLATING')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) # TODO
    for i in range(3):
        block = blocky_dataset["train"][i]
        block.pop("word_ids")
        data_collator([block])
        print(f"original block= {tokenizer.decode(block['labels'])}", escape_newlines=True)
        print(f"        masked= {tokenizer.decode(data_collator([block])['input_ids'][0])}", escape_newlines=True)
        
    print(f'\n=== FINETUNEING')
    model_name = "huggingface/CodeBERTa-small-v1"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    print(f"Model name   = {model_name}")
    print(f"Model params = {model.num_parameters() / 1_000_000:.1f}M")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"mlm-finetune",
            overwrite_output_dir=True,
            eval_strategy="epoch",
            learning_rate=1.5e-4,
            weight_decay=0.01,
            per_device_train_batch_size=TRAINING_BATCH_SIZE,
            per_device_eval_batch_size=TRAINING_BATCH_SIZE,
            fp16=True,
            logging_steps=len(blocky_dataset["train"]) // TRAINING_BATCH_SIZE,
            num_train_epochs=3, # 3
            # remove_unused_columns=False, # TODO
        ),
        train_dataset=blocky_dataset["train"],
        eval_dataset=blocky_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    train_results = trainer.train()
    print(f"Train time   = {train_results.metrics['train_runtime']:.1f}s")
    print(f"Train speed  = {train_results.metrics['train_samples_per_second']:.1f} samples/sec")
    eval_results = trainer.evaluate()
    print(f"Train loss   = {train_results.metrics['train_loss'] * 100:.2f}%")
    print(f"Eval loss    = {eval_results['eval_loss'] * 100:.2f}%")
    print(f"Perplexity   = {math.exp(eval_results['eval_loss']):.3f}")
    print( """--- WATERMARK
Model name   = huggingface/CodeBERTa-small-v1
Model params = 83.5M
Train time   = 10.3s
Train speed  = 620.1 samples/sec
Train loss   = 288.53%
Eval loss    = 139.88%
Perplexity   = 4.050
--- TO TRY
1. different model
1. different keywords
1. different params
1. headrel instead of pos
1. only leaf sentences
1. different optimizers
1. use code model to represent trees
""")

    print(f'\n=== PREDICTING')
    mask_filler = pipeline("fill-mask", model, tokenizer=tokenizer, device='cuda')

    with open('ar_nyuad-ud/trial6.csv', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = []
        for row in reader:
            rows.append(row)
        stats = defaultdict(lambda: defaultdict(lambda: 0))
        for cluster_id, prob, text, pos, sent_id, min_id, max_id, full_sent in tqdm(rows):
            full_sent_processed = [POS2WORD[w] if w in POS2WORD else w for w in full_sent.split(' ')]
            full_sent_processed_center = full_sent_processed.index('<mask>')
            full_sent_processed_start = max(
                0, full_sent_processed_center - TOKENIZATION_BLOCK_SIZE // 2)
            full_sent_processed_end = min(
                len(full_sent_processed), full_sent_processed_center + TOKENIZATION_BLOCK_SIZE // 2)
            full_sent_processed = ' '.join(full_sent_processed[full_sent_processed_start:full_sent_processed_end])
    
            print(full_sent)
            print(f"{cluster_id} {pos}  ->", newline=False)
            for pred in mask_filler(full_sent_processed, top_k=3):
                pred_token = pred['token_str']
                for k, v in POS2WORD.items():
                    pred_token = pred_token.replace(v, k)
                print(f"  [\'{pred_token}\' {pred['score'] * 100:.1f}%]", newline=False)

                stats[cluster_id][pred_token] += pred['score']
            print('')

    print(f'\n=== STATS ON CLUSTERS')
    stats = {
        cluster_id: sorted([(score / sum(v.values()), pred) for pred, score in v.items()], reverse=True)
        for cluster_id, v in stats.items()
    }
    for cluster_id, v in stats.items():
        print(f"{cluster_id}  ->", newline=False)
        for score, pred  in v:
            print(f"  [\'{pred}\' {score * 100:.1f}%]", newline=False)
        print('')

