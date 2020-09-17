# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import argparse
import logging
import os
import queue
import signal
import subprocess
import sys
import threading

import numpy as np
import os.path as osp
import sentencepiece as spm

# Set PATHs
PATH_TO_SENTEVAL = '..'
PATH_TO_DATA = osp.join('..', 'data')
PATH_TO_SKIPTHOUGHT = ''

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

proc = None
sp = None
outq = queue.Queue()

embedding_prefix = 'Sequence embedding: '

def prepare(params, samples):
    global proc, sp

    # keep open the extractor as a subprocess and send requests for each batch
    if proc is None:
        proc = subprocess.Popen(args.extract_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, encoding='utf8',
                                shell=True, bufsize=1, preexec_fn=os.setsid)

        t = threading.Thread(target=output_reader)
        t.start()

    if args.bpe_model and sp is None:
        sp = spm.SentencePieceProcessor(model_file=args.bpe_model)


def end_process():
    if proc is not None:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)


def output_reader():
    for line in proc.stdout:
        if line.startswith(embedding_prefix):
            outq.put(line)


def batcher(params, batch):
    for sentence_tokens in batch:
        if sp:
            sentence_tokens = sp.encode(' '.join(sentence_tokens), out_type=str)
        if len(sentence_tokens) > args.max_tokens:
            sentence_tokens = sentence_tokens[:args.max_tokens]
        sentence = ' '.join(sentence_tokens)
        proc.stdin.write(sentence + '\n')

    embeddings = []
    global count

    for i in range(len(batch)):
        line = outq.get(timeout=60)
        embeddings.append(list(map(float, line[len(embedding_prefix):].rstrip().split(' '))))

    return np.array(embeddings)


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-command', required=True,
                        help='Command to start the extractor that reads sentences from stdin '
                             'and prints embeddings to stdout')
    parser.add_argument('--bpe-model', help='Sentencepiece BPE to encode sentences before piping them to the command')
    parser.add_argument('--max-tokens', default=1000, type=int, help='Truncates sentences longer than this many tokens')
    args = parser.parse_args()

    try:
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion', 'MASC', 'BEAN']
        results = se.eval(transfer_tasks)
        print(results)
    except Exception as e:
        raise e
    finally:
        end_process()


