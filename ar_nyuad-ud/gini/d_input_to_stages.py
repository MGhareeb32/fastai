from util import print, conllu_corpus, tik, tok
from util import conllu_tree_clone, conllu_tree_draw
from c_stages import complexity_run, LVL_DEBUG, LVL_INFO, LVL_SILENT, rules_default
from g_input_to_ud import _analyze_pos, syntax_decode, build_tree
import csv
from collections import defaultdict
import ujson

def rules_load():
    rules = {}
    with open('ar_nyuad-ud/gini/c_stages.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, quotechar='|')
        for complexity, rule, *examples in reader:
            rules[rule] = {
                'complexity': int(complexity),
                'examples': examples,
            }
    return rules

def tree_draw(iteration_complexity, stage, sents):
    fname = f'ar_nyuad-ud/gini/png/stages{iteration_complexity}_{stage}'
    conllu_tree_draw(sents, font_name='Sakkal Majalla') \
        .render(filename=fname, format='png')
    print(f"{fname}.png")


def clean_complexity(sent_id, stuff):
    print('')
    print(sent_id)
    seen = set()
    for complexity, parts in stuff.items():
        print(f"complexity={complexity}")
        for i, part in enumerate(parts):
            if part in seen: continue
            part_formatted = part
            for j, s in enumerate(seen):
                if len(s.strip()) == 0: continue
                part_formatted = part_formatted.replace(s, f"({s.split(' ')[0]}...)")
            print(part_formatted)
        seen = set(parts)


if __name__ == '__main__':
    rules = rules_load()
    corpus = conllu_corpus()
    ud_trees = []

    for sent_id, txt in enumerate("""
أوعر الطريق بالمسافر
قال الرجل أنا شديد المرض و أرسل إلى المدير في العمل معتذرا عن التغيب ثم قام إلى السرير و نام
و غادر كنت مساء الأربعاء المدينة متوجها إلى ولاية أوهايو بعد أن استقل أحد باصات شركة سوني الشهيرة التي تجوب الولايات
""".split('\n')):
        if len(txt) == 0: continue

        print(f"\"{txt}\"")
        pos = list(_analyze_pos([txt]))[0]
        print(f"> {syntax_decode(pos)}")
        ud_trees.append(conllu_tree_clone(build_tree(txt, pos)))
        ud_trees[-1]['sent_id'] = f"TEST#{sent_id}"
        ud_trees[-1]['head'] = 0

    complexity_to_parts = defaultdict(lambda: defaultdict(lambda: set()))
    def print_stuff(iteration_complexity, stage, sents):
        def _find_plain(node, sent_id):
            [_find_plain(ch, sent_id) for ch in node['children']]
            if 'form_plain' in node:
                complexity_to_parts[sent_id][iteration_complexity].add(node['form_plain'])
        if stage != 'PUNCT': return
        for s in sents:
            _find_plain(s, s['sent_id'])

    tree_draw(0, '', ud_trees)
    apply_record = defaultdict(lambda: [])
    ud_trees = complexity_run(ud_trees, rules, rules_readonly=True,
                              apply_record=apply_record, verbosity=LVL_SILENT,
                              callback_fn=print_stuff)
    tree_draw(3, '', ud_trees)

    for sent_id, stuff in complexity_to_parts.items():
        clean_complexity(sent_id, stuff)
