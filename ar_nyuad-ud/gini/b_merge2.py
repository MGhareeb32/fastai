import pandas as pd
from util import print, conllu_corpus, tik, tok
from util import conllu_tree_clone, conllu_tree_draw
from collections import defaultdict
from copy import deepcopy
import imageio
import numpy as np

def merge2_name(node, parent):
    if parent is None or len(node['children']) > 0: return None

    if sum(node['id_range']) > sum(parent['id_range']):
        dist = node['id_range'][0] - parent['id_range'][1]
    else:
        dist = node['id_range'][1] - parent['id_range'][0]
    if abs(dist) > 1: return None

    if dist > 0:
        return f"{node['upos']}⟪{parent['upos']}"
    else:
        return f"{parent['upos']}⟫{node['upos']}"

BRACK_LT = "<FONT POINT-SIZE='14' COLOR='Red'><b>&#10219;</b></FONT>"
BRACK_RT = "<FONT POINT-SIZE='14' COLOR='Red'><b>&#10218;</b></FONT>"

# O(W)
def merge2_apply(sents, merge2s):

    def _merge2_apply(node, parent=None):
        if 'sent_id' in node:
            node = deepcopy(node)
            new_sents.append(node)
        # FIXME: special treatment of punct
        # if node['upos'] == 'PUNCT': node['upos'] = node['form']
        node['children'] = [ch for ch in node['children'] if _merge2_apply(ch, node)]

        merge2 = merge2_name(node, parent)
        if merge2 in merge2s:
            ns = sorted([parent, node], key=lambda n: n['id'])

            if parent['id'] < node['id']:
                parent['form'] = f"{node['form']}{BRACK_RT}{parent['form']}"
            else:
                parent['form'] = f"{parent['form']}{BRACK_LT}{node['form']}"
            parent['id_range'] = (ns[0]['id_range'][0], ns[1]['id_range'][1])
            parent['upos'] = merge2
            changes.append(merge2)
            return False
        return True 

    new_sents = []
    changes = []
    list(map(_merge2_apply, sents))
    return len(changes), new_sents

# O(W)
def merge2_find(sents, score_fn):
    def _node_stats(node, parent, depth, sent_id):
        assert(sent_id is not None)
    
        parent_dist = 0
        if parent is not None:
            parent_dist = min(max(node['id'] - parent['id'], -2), 2)

        return {
            'id': node['id'],
            'sent_id': sent_id,
            'depth^2': pow(depth, 2),
            'sibling_count': min(max(len(parent['children']) if parent is not None else 0, 1), 4),
            'slant': parent_dist if parent_dist > 0 else -parent_dist / 2,
        }

    def _merge2_find(node, parent=None, depth=0, sent_id=None):
        if sent_id is None: sent_id = node['sent_id']
        node_stats = _node_stats(node, parent, depth, sent_id)
    
        merge2 = merge2_name(node, parent)
        if merge2 is not None:
            merge2_stats[merge2].append(node_stats)
        tree_stats.append(node_stats)

        for ch in node['children']:
            _merge2_find(ch, node, depth+1, sent_id)

    tree_stats = []
    merge2_stats = defaultdict(lambda: [])
    list(map(_merge2_find, sents))
    tree_stats = pd.DataFrame(tree_stats)
    total_node_count = tree_stats['id'].count()
    total_sent_count = tree_stats.groupby('sent_id').mean()['id'].count()

    merge2_score = []
    for k, v in merge2_stats.items():
        stats = pd.DataFrame(v)
        node_count = stats['id'].count()
        sent_count = stats.groupby('sent_id').mean()['id'].count()
        score = (score_fn(
            k, stats, tree_stats,
            node_count / total_node_count,
            sent_count / total_sent_count,
        ) * 10000 // 10) / 1000
        merge2_score.append((score, k,))

    merge2_score = sorted(merge2_score, reverse=True)
    merge2_score = {k: v for v, k in merge2_score}
    return merge2_score

def image_png(fname, sents):
    conllu_tree_draw(sents, font_name='Sakkal Majalla') \
        .render(filename=fname, format='png')
    
def image_gif(prefix, sent_images):
    for sent_idx, paths in sent_images.items():
        pngs = [imageio.imread(f"{fname}.png") for fname in paths]
        pngs_size = np.max(np.array([im.shape for im in pngs]), axis=0)
        pngs_padded = []
        for im in pngs:
            sub = np.subtract(pngs_size, im.shape)
            pngs_padded.append(np.pad(
                im, ((0, sub[0]), (0, sub[1]), (0, sub[2])),
                mode='constant', constant_values=0,
            ))
        imageio.mimsave( f"ar_nyuad-ud/gini/png/sent_{prefix:02d}_{sent_idx:02d}.gif", pngs_padded, loop=0, fps=2)

if __name__ == '__main__':
    def _save_pngs():
        for sent_idx in [1, 7, 12]:
            if sents[sent_idx] == new_sents[sent_idx]: continue
            sent_images[sent_idx].append(f"{fname_root}_{sent_idx:02d}_{iter_count:02d}")
            image_png(sent_images[sent_idx][-1], [sents[sent_idx]])

    corpus = conllu_corpus()

    for score_fn_idx, score_fn in enumerate([
        lambda m2s, ss, tss, nc, sc: min(sc / .3, 1) * (ss['slant'] * ss['depth^2'] * (5 - ss['sibling_count'])).quantile(.05),
        lambda m2s, ss, tss, nc, sc: min(sc / .3, 1) * (ss['slant'] * ss['depth^2'] * (4.5 - ss['sibling_count'])).quantile(.025),
        # lambda m2s, ss, tss, nc, sc: min(sc / .3, 1) * (ss['slant'] * ss['depth^2']).quantile(.1),
        # lambda m2s, ss, tss, nc, sc: min(nc / .2, 1) * (ss['slant'] * ss['depth']).quantile(.1),
        # lambda m2s, ss, tss, nc, sc: min(nc / .2, 1) * (ss['slant'] * ss['depth']).mean(),

    ]):
        sents = list(map(conllu_tree_clone, corpus[:500]))
        sent_images = defaultdict(lambda: [])
        fname_root = f"ar_nyuad-ud/gini/png/tmp_{score_fn_idx:02d}"

        tik(f"==== {fname_root}.png")
        for iter_count in range(100):
            merge2_score = merge2_find(sents, score_fn)
            if len(merge2_score) == 0: break
            merge2s = list(merge2_score.keys())[:5]
            n_tokens = max([m.count('⟪') + m.count('⟫') for m in merge2s])
            if n_tokens >= 6: break
            merge_count, new_sents = merge2_apply(sents, merge2s)

            print(f"  {iter_count: >3}: {merge_count: >5,} <{n_tokens} x {[(name, merge2_score[name]) for name in merge2s]}")
            # _save_pngs()
            sents = new_sents
        tok()
        image_png(fname_root, [sents[i] for i in [1, 7, 12, 13, 16, 50]])
        # image_gif(score_fn_idx, sent_images)
