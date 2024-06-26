from util import print, conllu_corpus, conllu_tree_rules, conllu_sent_text, conllu_node_text, tik, tok
from tqdm import tqdm
from collections import defaultdict
import random
import csv

def find_rules(corpus):
    tik(f"\n==== FINDING GENERIC RULES")
    rule_count = 0
    for sent in tqdm(corpus):
        for tree, root in conllu_tree_rules(sent.to_tree()):
            if len(tree) <= 1:
                continue
            yield (
                conllu_sent_text(tree, 'rule'),
                f"{sent.metadata['sent_id']} {conllu_sent_text(tree, 'id')}",
                conllu_node_text(root, 'form')
            )
            rule_count += 1
    print(f"{rule_count:,} rules found.")
    tok()

if __name__ == '__main__':
    corpus = conllu_corpus()
    # corpus = corpus[:100]
    rule_to_sents = defaultdict(lambda: [])
    rule_example = {}
    for rule_text, subsent, example in find_rules(corpus):
        rule_to_sents[rule_text].append(subsent)
        n = len(rule_to_sents[rule_text])
        if random.random() * n <= 1 or\
                rule_text in rule_example and len(example) < len(rule_example[rule_text]) - 10:
            rule_example[rule_text] = example

    tik(f"\n==== SERIALIZING RULES")
    rule_to_sents = sorted([
        (-len(sent_and_root), rule, sent_and_root)
        for rule, sent_and_root in rule_to_sents.items()
    ])
    rule_to_sents = {
        rule: sent_and_root
        for _, rule, sent_and_root in rule_to_sents
    }
    print(f"{len(rule_to_sents):,} distinct rules found.")
    with open(__file__.replace('.py', '.csv'), 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for rule, subsents in tqdm(rule_to_sents.items()):
            row = [rule, rule_example[rule]]
            for subsent in subsents:
                row.append(subsent)
            writer.writerow(row)
    tok()

