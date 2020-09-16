# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
BEAN formality evaluation
Ellie Pavlick and Joel Tetreault. "An Empirical Analysis of Formality in Online Communication". TACL 2016.
'''

from __future__ import absolute_import, division, unicode_literals

import logging
import os.path as osp

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV


class BeanEval(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed

        logging.debug('***** Transfer task : BEAN formality classification *****\n\n')

        self.eval_data = self.loadFile(task_path)

    def do_prepare(self, params, prepare):
        samples = self.eval_data['X']
        return prepare(params, samples)

    def loadFile(self, task_path):
        X, y = [], []
        with open(osp.join(task_path, 'scores'), encoding='utf8', errors='ignore') as f_scores, \
                open(osp.join(task_path, 'bean-tokenized-sentences'), encoding='utf8', errors='ignore') as f_sentences:
            for line_score, line_sentence in zip(f_scores, f_sentences):
                score = float(line_score.split("\t")[0])
                tokens = line_sentence.split()
                X.append(tokens)
                y.append(score)

        return {'X': X, 'y': y}

    def run(self, params, batcher):
        embed = {}
        bsize = params.batch_size

        logging.info('Computing embedding')
        # Sort to reduce padding
        sorted_data = sorted(zip(self.eval_data['X'],
                                 self.eval_data['y']),
                             key=lambda z: (len(z[0]), z[1]))
        self.eval_data['X'], self.eval_data['y'] = map(list, zip(*sorted_data))

        embed['X'] = []
        for ii in range(0, len(self.eval_data['y']), bsize):
            batch = self.eval_data['X'][ii:ii + bsize]
            embeddings = batcher(params, batch)
            embed['X'].append(embeddings)
        embed['X'] = np.vstack(embed['X'])
        embed['y'] = np.array(self.eval_data['y'])
        logging.info('Computed embeddings')

        # like in Pavlick and Tetreault (cv with 10 folds and just return mean cv score)
        clf = RidgeCV(cv=10, scoring=lambda estimator, X, y: spearmanr(estimator.predict(X), y)[0]
                      ).fit(embed['X'], embed['y'])

        return {'spearman': clf.best_score_}
