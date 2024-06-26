from transformers import pipeline

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

# CONDENSE

MASK = '؟،،،؟'
def mask_sent(parts, i1, i2):
    masked_sent = (' '.join(parts[:i1]) + f' {MASK} ' + ' '.join(parts[i2:])).strip()
    removed_part = ' '.join(parts[i1:i2])
    return masked_sent, removed_part

def condense(sent, unmasker, classifier):
    parts = sent.split(' ')
    print(' '.join(parts))
    for i in range(len(parts)):
        masked_sent, _ = mask_sent(parts, i, i+1)
        guess = unmasker(masked_sent.replace(MASK, '[MASK]'))
        parts[i] = guess[0]['token_str']
    print(' '.join(parts))

    parts = sent.split(' ')
    best_seq, best_score = sent, classifier(sent)[0]['score']
    for j in range(len(parts)):
        for k in range(len(parts)):
            for i in range(len(parts)):
                masked_sent, _ = mask_sent(parts, i, i+1)
                guess = unmasker(masked_sent.replace(MASK, '[MASK]'))
            
                seq = guess[1]['sequence']
                score = classifier(seq)[0]['score']
                if score < best_score:
                    best_seq, best_score = seq, score
            print(best_score, best_seq)
            parts = best_seq.split(' ')
        

# MAIN

if __name__ == '__main__':
    sents = [
    'وبالطبع لم يكن من السهل عليه مواجهة كاميرات التلفزيون وعدسات المصورين وهو يصعد الباص .',
    ]

    print("\n=== MASKING OUT SENTENCES")
    model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth'
    # model_name = ''CAMeL-Lab/bert-base-arabic-camelbert-ca''
    # model_name = "akhooli/gpt2-small-arabic"
    unmasker = pipeline('fill-mask', model=model_name)
    classifier = pipeline("text-classification", model=model_name)

    for sent in sents:
        parts = sent.split(' ')
        LEN = 3
        for i in range(0, len(parts) - LEN + 1):
            masked_sent, removed_part = mask_sent(parts, i, i+LEN)
            print(masked_sent)
            print(removed_part)
            for guess in unmasker(masked_sent.replace(MASK, '[MASK]')):
                print(f"  [\'{guess['token_str']}\' {guess['score'] * 100:.1f}%]", newline=False)
            print('')
    
    print("\n=== CONDENSING")
    condense(sents[0], unmasker, classifier)
