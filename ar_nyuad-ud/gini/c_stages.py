from util import print, conllu_corpus, tik, tok
from util import conllu_tree_clone, conllu_tree_draw
from collections import defaultdict
from copy import copy, deepcopy
import csv

def sent_png(fname, sents):
    # sents = [sents[i] for i in [548, 709, 755, 789]]
    sents = [sents[i] for i in [1, 7, 12, 13, 16, 50]]
    # sents = [sents[i] for i in [1]]
    conllu_tree_draw(sents, font_name='Sakkal Majalla') \
        .render(filename=fname, format='png')

LVL_SILENT = 0
LVL_INFO = 1
LVL_DEBUG = 2

STAGE_TO_UPOS = {}
STAGE_TO_UPOS['NOUN'] = ['NOUN', 'PROPN', 'PRON', 'ADJ', 'ADV', 'NUM', 'DET', 'X']
STAGE_TO_UPOS['PART'] = ['ADP', 'PART', 'SCONJ', 'CCONJ', 'INTJ'] + STAGE_TO_UPOS['NOUN']
STAGE_TO_UPOS['VERB'] = ['VERB', 'AUX'] + STAGE_TO_UPOS['PART']
STAGE_TO_UPOS['PUNCT'] = None
COMPLEXITY_TO_COLOR = ['YellowGreen', 'DarkOrange', 'DarkRed']

BRACK_LT = "<FONT POINT-SIZE='12' COLOR='???'><b>❰</b></FONT>"
BRACK_RT = "<FONT POINT-SIZE='12' COLOR='???'><b>❱</b></FONT>"


def rules_default():
    return defaultdict(lambda: {
        'freq': 0,
        'examples': [],
    })

def node_count(node):
    return sum([node_count(ch) for ch in node['children']]) + 1

def node_merge(n1, n2, upos_whitelist, merge_color):
    if n1 is None or n2 is None: return None
    if upos_whitelist is not None:
        if n1.get('upos_orig', n1['upos']) not in upos_whitelist: return None
        if n2.get('upos_orig', n2['upos']) not in upos_whitelist: return None

    if n1 in n2['children']: n1, n2 = n1, n2
    if len(n2['children']) > 0: return None
    merge = copy(n1)

    pos_n2_minus_n1 = (sum(n2['id_range']) - sum(n1['id_range'])) / 2
    if pos_n2_minus_n1 > 0:
        pos_dist = n2['id_range'][0] - n1['id_range'][1]
        merge['upos'] = f"{n2['upos']}⟪{n1['upos']}"
        merge['form'] = f"{n2['form']}{BRACK_LT.replace('???', merge_color)}{n1['form']}"
        merge['form_plain'] = f"{n1.get('form_plain', n1['form'])} {n2.get('form_plain', n2['form'])}"
        merge['id_range'] = (n1['id_range'][0], n2['id_range'][1])
    else:
        pos_dist = n2['id_range'][1] - n1['id_range'][0]
        merge['upos'] = f"{n1['upos']}⟫{n2['upos']}"
        merge['form'] = f"{n1['form']}{BRACK_RT.replace('???', merge_color)}{n2['form']}"
        merge['form_plain'] = f"{n2.get('form_plain', n2['form'])} {n1.get('form_plain', n1['form'])}"
        merge['id_range'] = (n2['id_range'][0], n1['id_range'][1])
    if abs(pos_dist) > 1: return None

    merge['upos_orig'] = n1.get('upos_orig', n1['upos'])
    return merge

def rules_freq_print(rules, one_line=False):
    if sum([s['freq'] for s in rules.values()]) == 0:
        print(f"    > NONE")
        return

    if one_line:
        print('    >', newline=False)
        for rule, stats in rules.items():
            print(f" {stats['freq']:,}={rule}", newline=False)
        print('')
    else:
        for rule, stats in rules.items():
            print(f"    > {stats['freq']: >5,} x {rule}")

def collapse_freq_print(rules, one_line=False):
    if sum(rules.values()) == 0:
        print(f"    > NONE")
        return

    if one_line:
        print('    >', newline=False)
        for rule, count in rules.items():
            print(f" {count:,}={rule}", newline=False)
        print('')
    else:
        for rule, count in rules.items():
            print(f"    > {count: >5,} x {rule}")

def rules_freq_sort(rules_new, rules_old={}):
    rules_new = sorted([
            (stats['freq'], rule, stats) for rule, stats in rules_new.items()
            if rule not in rules_old
        ], reverse=True)
    return  {rule: stats for _, rule, stats in rules_new}

def rules_extract(sents, rules_old, rules_readonly, min_freq, complexity, upos_whitelist, merge_color, verbosity):

    def _count_and_extract(node, parent=None):
        if parent is not None:
            merge = node_merge(parent, node, upos_whitelist, merge_color)
            if merge is not None:
                rules_new[merge['upos']]['freq'] += 1
                if len(rules_new[merge['upos']]['examples']) == 0:
                    rules_new[merge['upos']]['examples'].append(merge.get('form_plain', merge['form']))
                rules_new[merge['upos']]['complexity'] = complexity

        for ch in node['children']:
            _count_and_extract(ch, node)

    if verbosity > LVL_INFO: print(f"  Finding >={min_freq} frequency rules...")
    rules_new = rules_default()
    list(map(_count_and_extract, sents))
    rules_new = rules_freq_sort(rules_new, rules_old)
    if verbosity > LVL_INFO: rules_freq_print(rules_new, True)
    rules_new = {r: s for r, s in rules_new.items() if s['freq'] >= min_freq and not rules_readonly}
    if verbosity > LVL_INFO: print(f"  Found {len(rules_new):,} new rules, aside from the old {len(rules_old):,} rules.")
    rules_old.update(rules_new)
    return rules_new

def rules_apply(sents, rules, upos_whitelist, complexity, apply_record, merge_color, verbosity):

    def _rules_apply(node, parent=None, sent_id=None):
        node = copy(node)
        if node['upos'] == 'PUNCT': node['upos'] = node['form']
        if parent is None:
            sents_new.append(node)
            sent_id = node['sent_id']
        # update children first
        node_children = []
        for ch in node['children']:
            ch_new = _rules_apply(ch, node, sent_id)
            if ch_new is None: continue
            node_children.append(ch_new)
        node['children'] = node_children
        # merge self with parent if needed
        merge = node_merge(parent, node, upos_whitelist, merge_color)
        if merge is not None and merge['upos'] in rules and rules[merge['upos']]['complexity'] <= complexity:
            if apply_record is not None:
                apply_record[sent_id].append((merge['upos'], merge['form_plain']))
            rules_freq[merge['upos']]['freq'] += 1
            for k in merge.keys(): parent[k] = merge[k]
            return None
        return node 

    if verbosity > LVL_INFO: print(f"  Applying {len(rules):,} rules...")
    rules_freq = rules_default()
    sents_new = []
    list(map(_rules_apply, sents))
    rules_freq = rules_freq_sort(rules_freq)
    if verbosity > LVL_INFO: rules_freq_print(rules_freq)
    return sents_new, rules_freq

def rules_extract_and_apply(
        sents, rules_old, rules_readonly, rule_freq_min, complexity, upos_whitelist, apply_record, merge_color, verbosity):

    if verbosity > LVL_INFO: print('')
    rules_new = rules_extract(sents, rules_old, rules_readonly, rule_freq_min, complexity, upos_whitelist, merge_color, verbosity)
    sents, rules_freq = rules_apply(sents, rules_old, upos_whitelist, complexity, apply_record, merge_color, verbosity)
    return sents, rules_freq, len(rules_new)

def merge_collapse(sents, verbosity):
    
    def _merge_collapse(node, parent=None):
        node = copy(node)
        if parent is None:
            sents_new.append(node)

        if 'upos_orig' in node:
            node['upos'] = node['upos_orig']
            collapse_freq[node['upos_orig']] += 1
            del node['upos_orig']
        node['children'] = [_merge_collapse(ch, node) for ch in node['children']]
        return node

    sents_new = []
    collapse_freq = defaultdict(lambda: 0)
    list(map(_merge_collapse, sents))
    if verbosity > LVL_INFO:
        print(f"  Collapsed {sum(collapse_freq.values()):,} merges...")
        collapse_freq_print(collapse_freq, True)
    return sents_new, collapse_freq

def stage_exhaust(
        stage_name, sents, rules, rules_readonly, rule_freq_frac, complexity, apply_record, merge_color, verbosity):

    if verbosity > LVL_SILENT: 
        print(f"\n  == STAGE '{stage_name}'. {sum(map(node_count, sents)):,} nodes...")
    rule_apply_count_total = 0

    for iteration_collapse in range(100 if not rules_readonly else 1):
        rules_new_count_total = 0
        rules_freq_total = rules_default()

        for iteration_rules in range(100):
            rule_freq_min = int(len(sents) / rule_freq_frac)
            sents, rules_freq, rules_new_count = rules_extract_and_apply(
                sents, rules,
                rules_readonly=rules_readonly,
                rule_freq_min=rule_freq_min,
                complexity=complexity,
                upos_whitelist=STAGE_TO_UPOS[stage_name],
                apply_record=apply_record,
                merge_color=merge_color,
                verbosity=verbosity)
            for rule, stats in rules_freq.items(): rules_freq_total[rule]['freq'] += stats['freq']
            rule_apply_count = sum([s['freq'] for _, s in rules_freq.items()])
            rule_apply_count_total += rule_apply_count

            if verbosity > LVL_INFO: print(f"  {rules_new_count:,} new rules (>={rule_freq_min}). {rule_apply_count:,} rule applications.")
            rules_new_count_total += rules_new_count
            if rule_apply_count == 0 and rules_new_count == 0: break

        sents, collapse_freq = merge_collapse(sents, verbosity)
        if sum(collapse_freq.values()) == 0: break

        if verbosity > LVL_SILENT: 
            print(f"  {rules_new_count_total: >2,} new rules (>={rule_freq_min: >4,}).", newline=False)
            print(f" {sum([s['freq'] for s in rules_freq_total.values()]): >3,} applications.", newline=False)
            print(f" {sum(collapse_freq.values()): >4,} merge collapses.", newline=False)
            rules_freq_print(rules_freq_total, True)
    return sents, rule_apply_count_total

def complexity_run(sents, rules, rules_readonly, apply_record, verbosity, callback_fn=None):
    def _print_progres(iteration_complexity):
        print(f"  {iteration_complexity: >2}]", newline=False)
        print(f"    nodes_per_sent={(sum(map(node_count, sents))/len(sents)):_>4.1f}", newline=False)
        print(f"    sents_left={sum([node_count(s) > 1 for s in sents]):_>6,}", newline=False)
        if iteration_complexity > -1:
            print(f"    rule_apply_count={rule_apply_count_total:_>6,}", newline=False)
            print(f"    new_rules={rule_count_end-rule_count_start:_>6,}", newline=False)
            print(f"    freq_frac={rule_freq_frac:_>6}", newline=False)
        print('')


    print(f"\n==== EXTRACTING RULES FROM {len(sents):,} SENTENCES = {sum(map(node_count, sents)):,} NODES...")
    _print_progres(-1)
    for iteration_complexity in range(3):
        if verbosity > LVL_SILENT:
            print(f"\n==== COMPLEXITY #{iteration_complexity}...")

        rule_count_start = len(rules)
        rule_freq_frac = pow(2, [3, 5, 8][iteration_complexity])
        rule_apply_count_total = 0

        merge_color = COMPLEXITY_TO_COLOR[iteration_complexity]
        for stage in ['NOUN', 'PART', 'VERB', 'PUNCT']:
            sents, rule_apply_count = stage_exhaust(
                stage, sents, rules, rules_readonly, rule_freq_frac,
                iteration_complexity, apply_record, merge_color, verbosity)
            rule_apply_count_total += rule_apply_count
            if callback_fn is not None: callback_fn(iteration_complexity, stage, sents)

        rule_count_end = len(rules)

        _print_progres(iteration_complexity)
        if sum(map(node_count, sents)) == len(sents): break

    return sents

if __name__ == '__main__':
    corpus = conllu_corpus()
    sents = list(map(conllu_tree_clone, corpus))
    rules = rules_default()
    sent_png('ar_nyuad-ud/gini/png/stages_01', sents)
    sents = complexity_run(sents, rules, False, None, LVL_SILENT)
    sent_png('ar_nyuad-ud/gini/png/stages_02', sents)

    print(f'==== TOTAL_RULES={len(rules)}')
    # sorted_rules = sorted([((s['complexity'], -s['freq']), r, s) for r, s in rules.items()])
    # sorted_rules = {r: s for _, r, s in sorted_rules}
    with open('ar_nyuad-ud/gini/c_stages.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, quotechar='|')
        for r, s in rules.items():
            writer.writerow([s['complexity'], r] + s['examples'])

"""
  -1]    nodes_per_sent=37.2    sents_left=_1,986
   0]    nodes_per_sent=14.4    sents_left=_1,964    rule_apply_count=45,412    new_rules=____43    freq_frac=_____8
   1]    nodes_per_sent=_7.1    sents_left=_1,272    rule_apply_count=14,479    new_rules=____57    freq_frac=____32
   2]    nodes_per_sent=_2.2    sents_left=___310    rule_apply_count=_9,625    new_rules=___255    freq_frac=___256
==== TOTAL_RULES=355

  -1]    nodes_per_sent=37.2    sents_left=_1,986
   0]    nodes_per_sent=20.6    sents_left=_1,973    rule_apply_count=33,019    new_rules=____18    freq_frac=_____4
   1]    nodes_per_sent=_7.1    sents_left=_1,272    rule_apply_count=26,872    new_rules=____95    freq_frac=____32
   2]    nodes_per_sent=_2.2    sents_left=___310    rule_apply_count=_9,625    new_rules=___251    freq_frac=___256
==== TOTAL_RULES=364
"""