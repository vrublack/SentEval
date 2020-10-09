import sys

# source: https://sources.debian.org/src/unidic-mecab/2.1.2~dfsg-2/right-id.def/
POS_TO_CONTEXT_ID = {
    '文末': 847,  # sentence ending
    '記号': 5977,  # symbol
    '名詞': 5146,  # noun
    '動詞': 891,  # verb
    '助動詞': 9,  # aux. verb
    '接頭詞': 5932,  # prefix
    '助詞': 661,  # particle
    '感動詞': 5687,  # interjection
    '連体詞': 5979,  # adnominal
    '形容詞': 5160,  # adjective
    '接続詞': 5930,  # conjunction
    '副詞': 4,  # adverb
    'フィラー': 5686,  # filler
}

POS_CONVERSION = {
    "フィラー": "感動詞",
    "接頭詞": "接頭辞",
    "文末": "助詞"
}


def main():
    all_entries = set()

    for i, line in enumerate(sys.stdin):
        if i == 0:
            # header
            continue

        if not line:
            break

        comps = line.split(',')
        entry1, entry2 = comps[0], comps[1]
        all_entries.add(entry1)
        all_entries.add(entry2)

    for entry in all_entries:
        word, pos = tuple(entry.split('/'))

        # using same context id for left and right
        context_id = POS_TO_CONTEXT_ID[pos]

        if pos in POS_CONVERSION:
            pos2 = pos
            pos = POS_CONVERSION[pos]
        else:
            pos2 = "一般"

        # choose lowest cost possible so it always gets chosen
        cost = 7000

        print(f"{word},{context_id},{context_id},{cost},{word},{pos},{pos2},*,*,*,*,{word},{word},*,*,*,*,*")


if __name__ == '__main__':
    main()
