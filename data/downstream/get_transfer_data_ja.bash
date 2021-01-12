#!/bin/bash

if [[ -z $CORPUS_JA ]]; then
  echo "CORPUS_JA variable must be set to point to a monolingual Japanese corpus"
  exit 1
fi

if [[ -z $MAIN_CODE_DIR ]]; then
  echo "MAIN_CODE_DIR variable must be set to point to the main codebase with the Japanese package"
  exit 1
fi

data_path=$(pwd)

# Amazon Japan
git clone https://github.com/Darkmap/japanese_sentiment
mkdir -p $data_path/AmazonJa
mv japanese_sentiment/data/* $data_path/AmazonJa
for file in $data_path/AmazonJa/*.txt; do
   # is already tokenized, detokenize (remove spaces), then tokenize with Mecab in SentEval task
   python3 -m Japanese.split_ja_sentences < $file | sed "s/ //g" > $file.sp
done
rm -r -f japanese_sentiment


# Rite2
mkdir -p $data_path/Rite2
curl -Lo $data_path/Rite2/RITE2_JA_bc-mc-unittest_forOpenAccess.tar.gz http://warehouse.ntcir.nii.ac.jp/openaccess/rite/data/RITE2_JA_bc-mc-unittest_forOpenAccess.tar.gz
(cd $data_path/Rite2 && tar -xvf $data_path/Rite2/RITE2_JA_bc-mc-unittest_forOpenAccess.tar.gz)
for unneeded in $data_path/Rite2/*/*.parsed.*.xml; do
  rm $unneeded
done
rm $data_path/Rite2/RITE2_JA_bc-mc-unittest_forOpenAccess.tar.gz


# FormalityJa
olddir=$(pwd)
cd $MAIN_CODE_DIR || exit
python3 -m Japanese.create_classifier_dataset \
    --in-jp $CORPUS_JA \
    --out-dir $data_path/FormalityJa \
    --balance \
    --first 10000 \
    --task formality \
    --limit-substr "ありがとうございます" "おはようございます" \
    --deduplicate
cd $olddir || exit

# Akama style annotations
mkdir -p $data_path/StyleSimJa
git clone https://github.com/jqk09a/stylistic-word-similarity-dataset-ja
# we need to custommize tokenization because the Akama annotations used a tokenizer with a custom dictionary
python3 -m Japanese.create_sudachi_dict < stylistic-word-similarity-dataset-ja/stylistic_wordsim.csv > $data_path/StyleSimJa/sudachi_dict.csv
git clone https://github.com/WorksApplications/SudachiPy
sudachipy ubuild $data_path/StyleSimJa/sudachi_dict.csv -o $data_path/StyleSimJa/sudachi_dict.dic
echo "{" > SudachiPy/sudachipy/resources/sudachi-custom.json
echo "\"userDict\" : [\"$data_path/StyleSimJa/sudachi_dict.dic\"]," >> SudachiPy/sudachipy/resources/sudachi-custom.json
sed 1d SudachiPy/sudachipy/resources/sudachi.json >> SudachiPy/sudachipy/resources/sudachi-custom.json
python3 -m Japanese.filter_corpus_akama \
  --akama-file stylistic-word-similarity-dataset-ja/stylistic_wordsim.csv \
  --corpus $CORPUS_JA \
  --sudachipy-config SudachiPy/sudachipy/resources/sudachi-custom.json \
  --out-path-dev $data_path/StyleSimJa/stylistic_sentsim_dev.csv \
  --out-path-test $data_path/StyleSimJa/stylistic_sentsim_test.csv
rm -r -f stylistic-word-similarity-dataset-ja
rm -r -f SudachiPy

# remove moses folder
rm -rf mosesdecoder


# Japanese word probing
python3 -m Japanese.create_word_content_dataset \
    --corpus $CORPUS_JA \
    --out-path ../probing/word_content_japanese.txt
