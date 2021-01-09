import csv
import logging
import math
import os.path as osp

import MeCab
import numpy as np
from scipy import spatial
from scipy.stats import spearmanr


class StyleSimJaEval:
    def __init__(self, task_path):
        self.mecab_wrapper = MeCab.Tagger("-Owakati")

        self.sents = {}
        for sp in ['dev', 'test']:
            sent1, sent2, sim = self.load_file(osp.join(task_path, f'stylistic_sentsim_{sp}.csv'))
            self.sents[sp] = {'1': sent1, '2': sent2, 'sim': sim}

    def load_file(self, sentence_style_path):
        sent1, sent2, sim = [], [], []
        with open(sentence_style_path, encoding='utf8') as f:
            reader = csv.reader(f)
            next(reader)  # skip the header
            for row in reader:
                sent1.append(self.tokenize(row[0]))
                sent2.append(self.tokenize(row[1]))
                sim.append(float(row[4]))

        return sent1, sent2, sim

    def tokenize(self, sentence):
        return self.mecab_wrapper.parse(sentence).split()

    def do_prepare(self, params, prepare):
        samples = []
        for sp in ['dev', 'test']:
            samples += self.sents[sp]['1'] + self.sents[sp]['2']
        prepare(params, samples)

    def run(self, params, batcher):
        results = {}

        for sp in ['dev', 'test']:
            embed = {}
            bsize = params.batch_size

            logging.info('Computing embeddings')
            # Sort to reduce padding
            sorted_data = sorted(zip(self.sents[sp]['1'],
                                     self.sents[sp]['2'],
                                     self.sents[sp]['sim']),
                                 key=lambda z: (len(z[0]), len(z[1])))
            sents1, sents2, sims = map(list, zip(*sorted_data))
            sents = {'1': sents1, '2': sents2}

            for key in sents.keys():
                embed[key] = []
                for ii in range(0, len(sents[key]), bsize):
                    batch = sents[key][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    embed[key].append(embeddings)
                embed[key] = np.vstack(embed[key])
                logging.info('Computed {0} embeddings'.format(key))

            embed_dist = []
            sims_keep = []
            for e1, e2, sim in zip(embed['1'], embed['2'], sims):
                dist = spatial.distance.cosine(e1, e2)
                if math.isnan(dist):
                    print('Warning: skipped nan value')
                else:
                    embed_dist.append(1 - dist)
                    sims_keep.append(sim)

            results.update({f'spearman_{sp}': spearmanr(embed_dist, sims_keep)[0], f'n_{sp}': len(sims_keep)})

        return results

