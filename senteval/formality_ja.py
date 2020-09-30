# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import absolute_import, division, unicode_literals

import logging
import os

import MeCab
import numpy as np
from sklearn.model_selection import train_test_split

from senteval.tools.validation import KFoldClassifier


class FormalityJaEval(object):
    def __init__(self, task_path, nclasses=3, seed=1111):
        self.seed = seed

        self.nclasses = nclasses
        logging.debug('***** Transfer task : FormalityJa *****\n\n')

        self.mecab_wrapper = MeCab.Tagger("-Owakati")

        X_all = self.load_sentences(os.path.join(task_path, 'sentences.txt'))
        y_all = self.load_labels(os.path.join(task_path, 'formality-labels.txt'))

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=seed)

        self.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

    def do_prepare(self, params, prepare):
        samples = self.data['train']['X'] + \
                  self.data['test']['X']
        return prepare(params, samples)

    def tokenize(self, sentence):
        return self.mecab_wrapper.parse(sentence).split()

    def load_sentences(self, fpath):
        with open(fpath, encoding='utf8') as f:
            return list(map(self.tokenize, f.read().strip().split('\n')))

    def load_labels(self, fpath):
        label2idx = {}
        y = []
        with open(fpath, encoding='utf8') as f:
            for label in f:
                label = label.strip()
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
                y.append(label2idx[label])
        assert len(label2idx) == self.nclasses
        return y

    def run(self, params, batcher):
        embed = {'train': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.data[key]['X'],
                                     self.data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.data[key]['X'], self.data[key]['y'] = map(list, zip(*sorted_data))

            embed[key]['X'] = []
            for ii in range(0, len(self.data[key]['y']), bsize):
                batch = self.data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                embed[key]['X'].append(embeddings)
            embed[key]['X'] = np.vstack(embed[key]['X'])
            embed[key]['y'] = np.array(self.data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}

        clf = KFoldClassifier(embed['train'], embed['test'], config_classifier)

        devacc, testacc, _ = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            FormalityJa classification\n'.format(devacc, testacc))

        return {'devacc': devacc, 'acc': testacc,
                'ntest': len(embed['test']['X'])}
