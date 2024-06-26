
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from util import conllu_corpus, print, tik, tok

def preprocess_function(examples):
    inputs = [example['ar'] for example in examples["translation"]]
    targets = [example['en'] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

LETTER2UPOS = {
    'P': 'PROPN',
    'N': 'NOUN',
    'O': 'PRON',
    'R': 'VERB',
    'X': 'AUX',
    'J': 'ADJ',
    'V': 'ADV',
    'L': 'ADP',
    'T': 'PART',
    '.': 'PUNCT',
    '#': 'NUM',
    'S': 'SCONJ',
    'C': 'CCONJ',
    'D': 'DET',
    '?': 'X',
    'I': 'INTJ',
}
UPOS2LETTER = {v: k for k, v in LETTER2UPOS.items()}
LETTER2DEPREL = {
    'r': 'root',
    'q': 'case',
    'y': 'mark',
    'f': 'flat',
    'd': 'compound',
    'e': 'cc',
    'w': 'cop',
    'b': 'conj',
    'h': 'appos',
    'l': 'acl',
    'k': 'advcl',
    'z': 'discourse',
    't': 'det',
    'j': 'dep',
    'u': 'aux',
    'c': 'ccomp',
    'x': 'xcomp',
    'p': 'punct',
    's': 'nsubj',
    'o': 'obj',
    'i': 'iobj',
    'm': 'nmod',
    'a': 'amod',
    'v': 'advmod',
    'n': 'nummod',
    '\'': 'nmod:poss',
}
DEPREL2LETTER = {v: k for k, v in LETTER2DEPREL.items()}

def syntax_decode(pos):
    ans = []
    for t in pos.split(' '):
        ans.append(f"{LETTER2UPOS[t[0]]}{LETTER2DEPREL[t[1]]}{t[2:]}")
    return ' '.join(ans)

def generate_dataset():
    corpus = conllu_corpus(['ar_nyuad-ud/dev.conllu', 'ar_nyuad-ud/train.conllu'])
    
    def token_depth(sent, t):
        if t['head'] == 0:
            return 0
        return 1 + token_depth(sent, sent.filter(id=t['head'])[0])

    def sentence_syntax(sent):
        res = []
        for t in sent:
            if t['upos'] == '_': continue
            upos = UPOS2LETTER[t['upos']]
            deprel = DEPREL2LETTER[t['deprel']]
            depth = token_depth(sent, t)
            parent_dist = t['head'] - t['id']
            res.append(f"{upos}{deprel}{depth}{'!' if parent_dist > 0 else ''}")
        return ' '.join(res)

    for sent in corpus:
        yield {
            'translation': {
                "ar": sent.metadata['text'],
                "en": sentence_syntax(sent)
            }
        }

if __name__ == '__main__':
    ds = Dataset.from_generator(generate_dataset)
    tik(f"\n==== GENERATING TRAIN-TEST SPLIT")
    ds = ds.train_test_split()
    tok()
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    tokenized_ds = ds.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch", # epoch
        learning_rate=1e-4, # 2e-5
        per_device_train_batch_size=32, # 16
        per_device_eval_batch_size=32, # 16
        weight_decay=0.01, # 0.01
        save_total_limit=3, # 3
        num_train_epochs=4, # 3
        fp16=True, # True
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()

    for txt in """
الحمد لله رب العالمين
يتزعم ذلك أحمد فريد
حضرن اللقاء
قال الرجل أنا شديد المرض
و نادى الرئيس بضرورة الاتحاد
أرسل إلى المدير يسأل عن الراتب
قام إلى البيت ينظفه
""".split('\n'):
        if len(txt) == 0: continue
        pos = tokenizer.decode(
            model.generate(
                tokenizer(txt, return_tensors="pt").input_ids.cuda(),
                max_new_tokens=128, do_sample=True, top_k=30, top_p=0.95).flatten(),
            skip_special_tokens=True)

        print(txt)
        print(pos)
        print(syntax_decode(pos))
        print('')