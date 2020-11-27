import argparse
import collections
import random

from sudachipy import dictionary

random.seed(12345)


def generate_corpus_tokens(corpus_path, return_detokenized=False):
    tokenizer_obj = dictionary.Dictionary().create()

    with open(corpus_path, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            toks = list(map(lambda x: x.surface(), tokenizer_obj.tokenize(line)))
            if return_detokenized:
                yield toks, line
            else:
                yield toks


def estimate_token_frequencies(corpus_path):
    token_freq = collections.defaultdict(int)

    # estimate token frequencies
    for i, line in enumerate(generate_corpus_tokens(corpus_path)):
        if i >= 100000:
            break
        for tok in line:
            token_freq[tok] += 1

    freq_sorted = sorted([(freq, tok) for tok, freq in token_freq.items()], reverse=True)

    return [tok for freq, tok in freq_sorted]


def train_dev_test_split(lines, n_dev, n_test):
    lines_train = lines[:-(n_dev + n_test)]
    lines_dev = lines[-(n_dev + n_test):-n_test]
    lines_test = lines[-n_test:]

    assert len(lines_train) + len(lines_dev) + len(lines_test) == len(lines) \
           and len(lines_dev) == n_dev and len(lines_test) == n_test

    return lines_train, lines_dev, lines_test


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', required=True, help='Untokenized Japanese corpus')
parser.add_argument('--out-path', required=True)
parser.add_argument('--n-words', type=int, default=1000)
parser.add_argument('--sentences-per-word', type=int, default=120)
parser.add_argument('--n-dev', type=int, default=10000)
parser.add_argument('--n-test', type=int, default=10000)
args = parser.parse_args()

freq_sorted = estimate_token_frequencies(args.corpus)

# pick tokens with mid frequency rank (like https://arxiv.org/pdf/1805.01070.pdf)

selected_tokens = set(freq_sorted[2000:2000 + args.n_words])  # TODO inspect
token_sentences = collections.defaultdict(list)
complete = 0

for line, detok in generate_corpus_tokens(args.corpus, True):
    contained_tok = None
    for tok in line:
        if tok in selected_tokens:
            if contained_tok is None:
                contained_tok = tok
            else:
                # line should only contain one occurrence of one selected token
                contained_tok = None
                break

    if contained_tok is not None and len(token_sentences[contained_tok]) < args.sentences_per_word:
        token_sentences[contained_tok].append(detok)
        if len(token_sentences[contained_tok]) == args.sentences_per_word:
            complete += 1
            if complete == len(selected_tokens):
                break

token_sentences = list(sorted(token_sentences.items()))

flattened = []
for tok, sentences in token_sentences:
    for sent in sentences:
        flattened.append((tok, sent))
del token_sentences

if len(flattened) < args.n_words * args.sentences_per_word:
    print(f'Warning: not enough data to make {args.n_words * args.sentences_per_word} examples')

random.shuffle(flattened)
train, dev, test = train_dev_test_split(flattened, args.n_dev, args.n_test)

with open(args.out_path, 'w', encoding='utf8') as f:
    for sp, sp_name in [(train, 'tr'), (dev, 'va'), (test, 'te')]:
        for tok, sent in sp:
            tok = tok.replace("\t", "")
            sent = sent.replace("\t", "")
            f.write(f'{sp_name}\t{tok}\t{sent}\n')
