from __future__ import absolute_import, division, unicode_literals

import collections
import logging
import os
import xml.etree.ElementTree as ET

import JapaneseTokenizer
import numpy as np

from senteval.tools.validation import InnerKFoldClassifier


class Rite2JaBCEntailmentEval:
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : Rite2JaBC-Entailment*****\n\n')
        self.seed = seed
        self.mecab_wrapper = JapaneseTokenizer.MecabWrapper('unidic')
        dev = self.loadFile(os.path.join(task_path, 'RITE2_JA_dev_bc', 'RITE2_JA_dev_bc.xml'))
        test = self.loadFile(os.path.join(task_path, 'RITE2_JA_testlabel_bc', 'RITE2_JA_testlabel_bc.xml'))
        train = {}
        for key in dev.keys():
            train[key] = dev[key]
            train[key].extend(test[key])
        # use dev as test together to do cross validation
        self.data = {'train': train}

    def tokenize(self, sentence):
        return list(map(lambda tok: tok.word_surface, self.mecab_wrapper.tokenize(sentence).tokenized_objects))

    def loadFile(self, fpath):
        label2id = {'Y': 0, 'N': 1}
        data = {'X_A': [], 'X_B': [], 'y': []}

        tree = ET.parse(fpath)
        root = tree.getroot()

        for pair in root:
            data['y'].append(pair.attrib['label'])
            data['X_A'].append(self.tokenize(pair[0].text))
            data['X_B'].append(self.tokenize(pair[1].text))

        data['y'] = [label2id[s] for s in data['y']]
        return data

    def do_prepare(self, params, prepare):
        samples = self.data['train']['X_A'] + \
                  self.data['train']['X_B']
        return prepare(params, samples)

    def run(self, params, batcher):
        embed = {'train': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.data[key]['X_A'],
                                       self.data[key]['X_B'],
                                       self.data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                embed[key][txt_type] = []
                for ii in range(0, len(self.data[key]['y']), bsize):
                    batch = self.data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    embed[key][txt_type].append(embeddings)
                embed[key][txt_type] = np.vstack(embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        trainA = embed['train']['X_A']
        trainB = embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.data['train']['y'])

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid}
        clf = InnerKFoldClassifier(X=trainF, y=trainY, config=config)

        count = collections.defaultdict(int)
        for y in trainY:
            count[y] += 1
        logging.debug('Train histogram: ' + str(count))

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
                       Rite2JaBC-Entailment\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc}
