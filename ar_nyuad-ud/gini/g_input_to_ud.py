from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from util import print, conllu_tree_rules, conllu_sent_text, conllu_node_text
import re
import ujson, csv
from f_input_to_pos import LETTER2DEPREL, LETTER2UPOS, syntax_decode

_tokenizer = None
_model = None
def _load_model(path="results/checkpoint-1500"):
    global _tokenizer
    global _model
    if _model is None and _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(path)
        _model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return _tokenizer, _model

def _analyze_pos(texts):
    tokenizer, model = _load_model()
    for txt in texts:
        yield tokenizer.decode(
            model.generate(
                tokenizer(txt, return_tensors="pt").input_ids,
                max_new_tokens=128, do_sample=True, top_k=30, top_p=0.95).flatten(),
            skip_special_tokens=True)

def _split_pos(pos):
    upos = LETTER2UPOS[pos[0]]
    deprel = LETTER2DEPREL[pos[1]]
    depth = int(pos[2])
    depfwd = len(pos) > 3
    return (upos, deprel, depth, depfwd)

def _split_text(txt, pos):
    txt = txt.split(' ')
    pos = pos.split(' ')
    for i, p in enumerate(pos):
        upos, deprel, depth, depfwd = _split_pos(p)
        if upos == 'ADP' and len(txt) < len(pos):
            txt.insert(i+1, txt[i][1:])
            txt[i] = f"{txt[i][0]}"
        yield({
            'id': i+1,
            'form': txt[i],
            'upos': upos,
            'deprel': deprel,
            'depth': int(depth),
            'depfwd': depfwd,
            'children': [],
        })

def _build_tree(tokens, root):
    # go right
    i = root['id']-1
    while i > 0:
        i -= 1
        rch = tokens[i]
        if rch['depth'] == root['depth'] + 1 and 'depfwd' in rch and rch['depfwd']:
            root['children'].append(rch)
            rch['head'] = root['id']
        elif rch['depth'] > root['depth']:
            continue
        else:
            break
    root['children'].reverse()
    # go left
    i = root['id']-1
    while i < len(tokens) - 1:
        i += 1
        lch = tokens[i]
        if lch['depth'] == root['depth'] + 1 and 'depfwd' in lch and not lch['depfwd']:
            root['children'].append(lch)
            lch['head'] = root['id']
        elif lch['depth'] > root['depth']:
            continue
        else:
            break
    # recurse
    for ch in root['children']:
        _build_tree(tokens, ch)
    del root['depfwd']

    # for t in tokens:
    #     if t['depth'] == root['depth'] + 1:
    #         root['children'].append(t)

def build_tree(txt, pos):
    tokens = list(_split_text(txt, pos))
    root = min(tokens, key=lambda t: t['depth'])
    _build_tree(tokens, root)
    return root

def rules_load():
    rule2ex = {}
    with open('ar_nyuad-ud/gini/a_corpus_to_generic_rules.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for rule, ex, *instances in reader:
            rule2ex[rule] = ex
    return rule2ex

if __name__ == '__main__':
    rule2ex = rules_load()

    for txt in  """
يقول ذلك جمال عبد الناصر
جئن الساحة باكرا
قال الرجل أنا شديد المرض
و نادى الرئيس بضرورة الاتحاد
أرسل إلى المدير يسأل عن الراتب
قام إلى الباب و انصرف
""".split('\n'):
        if len(txt) == 0: continue

        print(f"\"{txt}\"")
        pos = list(_analyze_pos([txt]))[0]
        print(f"> {syntax_decode(pos)}")
        ud_tree = build_tree(txt, pos)
        # print(ujson.dumps(ud_tree, ensure_ascii=False, indent=2))
        for rule, node in conllu_tree_rules(ud_tree):
            txt_part = conllu_node_text(node)
            rule_text = conllu_sent_text(rule)
            if rule_text in rule2ex:
                print(f"- {rule_text}:")
                print(f"    {txt_part} e.g: {rule2ex[rule_text]}")
        print('')
