import argparse
import csv

from sudachipy import dictionary

from .create_sudachi_dict import POS_CONVERSION


def convert_pos(entry):
    word, pos = tuple(entry.split('/'))
    if pos in POS_CONVERSION:
        pos = POS_CONVERSION[pos]
    return word + '/' + pos


def main(args):
    with open(args.akama_file, encoding='utf8') as f:
        style_lines = f.read().rstrip().split('\n')
        header = style_lines[0]
        del style_lines[0]

    entry_to_sents = {}

    for line in style_lines:
        comps = line.split(',')
        entry_to_sents[comps[0]] = []
        entry_to_sents[comps[1]] = []

    tokenizer_obj = dictionary.Dictionary(args.sudachipy_config).create()

    with open(args.corpus, encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if i % 1000 == 0:
                print(f'Processing line {i}')
            morphemes = tokenizer_obj.tokenize(line)
            for m in morphemes:
                all_pos = m.part_of_speech()
                for pos_i in range(min(2, len(all_pos)) - 1, -1, -1):
                    entry = m.surface() + '/' + all_pos[pos_i]
                    if entry in entry_to_sents:
                        entry_to_sents[entry].append(line)
                        break

    for l in entry_to_sents.values():
        # keep the shortest sentences
        l.sort(key=lambda s: len(s))
        if len(l) > args.sentences_per_pair:
            del l[args.sentences_per_pair:]

    found, not_found = 0, 0
    total_list_len = 0
    for key, l in entry_to_sents.items():
        if l:
            found += 1
            total_list_len += len(l)
        else:
            not_found += 1

    print(f'Found {found} / {found + not_found} entries, avg list len {total_list_len / found}')

    found_pairs = 0
    with open(args.out_path, 'w', encoding='utf8') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['sentence 1', 'sentence 2'] + header.split(','))

        for line in style_lines:
            comps = line.split(',')
            entry1, entry2 = comps[0], comps[1]
            found = False
            for entry1_sent in entry_to_sents[entry1]:
                for entry2_sent in entry_to_sents[entry2]:
                    found = True
                    wr.writerow([entry1_sent] + [entry2_sent] + [entry1] + [entry2] + comps[2:])

            if found:
                found_pairs += 1

    print(f'Found {found_pairs} / {len(style_lines)} entry pairs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--akama-file', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--sudachipy-config', required=True)
    parser.add_argument('--sentences-per-pair', default=5, type=int, help='Finds up to this many sentences per pair')
    main(parser.parse_args())
