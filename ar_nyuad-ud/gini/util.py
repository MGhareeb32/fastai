from timeit import default_timer
from graphviz import Digraph, nohtml
import conllu
from copy import deepcopy

# LOGGING

_print_file = None
def print(*text, newline=True, escape_newlines=False):
    global _print_file
    if _print_file is None:
        _print_file = open(__file__.replace('.py', '.log'), 'w', encoding='utf-8')
    for t in text:
        txt = str(t)
        if escape_newlines: txt = txt.replace('\n', '\\n')
        _print_file.write(txt)

        if newline: _print_file.write('\n')
    _print_file.flush()

_tiktok = None

def tik(txt):
    global _tiktok
    _tiktok = default_timer()
    print(txt)

def tok():
    global _tiktok
    print(f"Elapsed time = {default_timer() - _tiktok:.2f}s")
    _tiktok = None

# CONLLU

DEPREL2AR = {
  'ccomp': 'مفل',
  'xcomp': 'مفص',
  'nsubj': 'فاع',
  'obj': 'مف1',
  'iobj': 'مف2',
  'nmod': 'ظرف',
  'advmod': 'حال',
  'amod': 'صفة',
  'nmod:poss': 'إضف',
  'nummod': 'رقم',
  'cop': 'فعل',
  'flat': 'بدل',
  'compound': 'ركب',
}

def conllu_corpus(paths=['ar_nyuad-ud/dev.conllu']):
    tik(f"\n==== READING CORPUS {paths}")
    corpus = []
    for path in paths:
        with open(path, 'r') as file:
            corpus.extend(conllu.parse(file.read()))
    print(f"{len(corpus):,} sentences loaded.")
    tok()
    return corpus

def _conllu_tree_draw(g, subgraph_idx, sent_id, node, depth):
    id_self = f"{subgraph_idx}|{sent_id}|{node['id']}"
    id_head = f"{subgraph_idx}|{sent_id}|{node['head']}"
    pos_x = -sum(node['id_range']) * .3
    pos_y = -depth * 1.5
    g.node(
        id_self,
        label=f"<<b>{node['form']}<br /><FONT POINT-SIZE='30' COLOR='DarkGreen'>{node['upos']}</FONT></b>>",
        pos=f"{pos_x},{pos_y}!")
    if not id_head.endswith('|0'):
        g.edge(id_head, id_self, headlabel=f"<<b>{node['deprel']}</b>>")
    
    max_depth = depth
    for ch in node['children']:
        max_depth = max(max_depth, _conllu_tree_draw(g, subgraph_idx, sent_id, ch, depth+1))
    return max_depth

def conllu_tree_draw(sentences, font_name='Times-Roman'):
    g = Digraph('G', engine="fdp",
                graph_attr={'fontname': font_name, 'splines': 'ortho', 'overlap': 'vpsc'},
                node_attr={'fontname': font_name, 'fontsize': '34', 'shape': 'plain'},
                edge_attr={'fontname': font_name, 'fontsize': '32', 'fontcolor': 'CornflowerBlue', 'color': 'LightPink', 'arrowsize': '.5', 'dir': 'back', 'penwidth': '1'})
    # for sent in sentences:
    #     print(sent.metadata['text'])
    depth = 0
    for subgraph_idx, sent in enumerate(sentences):
        sent_id = sent['sent_id'].replace(':', '#')
        with g.subgraph(name=sent_id) as c:
            depth = _conllu_tree_draw(c, subgraph_idx, sent_id, sent, depth) + 1
    return g


def conllu_tree_clone(node):
  sent_id = None
  if type(node) == conllu.TokenList:
    sent_id = node.metadata['sent_id']
    node = node.to_tree()
  if type(node) == conllu.TokenTree:
    node_dict = {
      'id': node.token['id'],
      'id_range': (node.token['id'], node.token['id']),
      'head': node.token['head'],
      'form': node.token['form'],
      'lemma': node.token['lemma'],
      'upos': node.token['upos'],
      'deprel': node.token['deprel'],
      'children': [conllu_tree_clone(ch) for ch in node.children],
    }
  else:
    node_dict = {
      'id': node['id'],
      'id_range': (node['id'], node['id']),
      'head': node.get('head', 0),
      'form': node['form'],
      'lemma': node.get('lemma', 'unk'),
      'upos': node['upos'],
      'deprel': node['deprel'],
      'children': [conllu_tree_clone(ch) for ch in node['children']],
    }
  if sent_id is not None: node_dict['sent_id'] = sent_id
  return node_dict

def conllu_node_clone(node):
  return deepcopy(node)


def conllu_sent_text(sent, key='rule'):
  return ' '.join([str(t[key]) for t in sent])

def conllu_node_text(node, key='form'):
  def _conllu_node_text(node, output):
    sent = [node] + node['children']
    sent = sorted(sent, key=lambda t: t['id'])
    for ch in sent:
      if ch == node:
         output.append(node)
      else:
        _conllu_node_text(ch, output)
     
  ans = []
  _conllu_node_text(node, ans)
  return ' '.join([str(t[key]) for t in ans])

def conllu_node_rule(root):
  def _conllu_node_rule(node):
    node = conllu_node_clone(node)
    node['rule'] = f"<{node['deprel']}>"
    if node['deprel'] == 'root':
      node['rule'] = f"[{node['upos']}]"
    return node

  root['deprel'] = 'root'
  flat_sent = list(map(_conllu_node_rule, [root] + root['children']))
  flat_sent = sorted(flat_sent, key=lambda t: t['id'])
  return flat_sent

def conllu_tree_rules(sent):
  def _conllu_tree_rules(node, output):
    flat_sent = conllu_node_rule(node)
    output.append((flat_sent, node))
    for t in flat_sent:
      if t['rule'].startswith('<'):
        _conllu_tree_rules(t, output)
    
  output = []
  _conllu_tree_rules(sent, output)
  return output

if __name__ == '__main__':
    corpus = conllu_corpus()
    print(corpus[0].metadata['text'])
    print(conllu_node_text(corpus[0].to_tree(), 'form'))