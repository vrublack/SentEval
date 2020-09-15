# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import argparse
import logging
import subprocess
import sys
import numpy as np
import os.path as osp


# Set PATHs
PATH_TO_SENTEVAL = '..'
PATH_TO_DATA = osp.join('..', 'data')
PATH_TO_SKIPTHOUGHT = ''

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

proc = None


def prepare(params, samples):
    global proc
    # keep open the extractor as a subprocess and send requests for each batch
    proc = subprocess.Popen(args.extract_command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, encoding='utf8',
                            shell=True, bufsize=1)


def batcher(params, batch):
    for sentence in batch:
        proc.stdin.write(' '.join(sentence) + '\n')

    embeddings = []
    received_embeddings = 0
    embedding_prefix = 'Sequence embedding: '
    for line in proc.stdout:
        if line.startswith(embedding_prefix):
            embeddings.append(list(map(float, line[len(embedding_prefix):].rstrip().split(' '))))
            received_embeddings += 1
            if received_embeddings == len(batch):
                break

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
    args = parser.parse_args()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
