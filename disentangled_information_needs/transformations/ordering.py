from textattack.augmentation import Augmenter
from textattack.transformations import *
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 

class OrderingActions():
    def __init__(self, queries):
        self.queries = queries
        self.augmenters = [
            Augmenter(transformation=WordInnerSwapRandom(), transformations_per_example=1)            
        ]

    def shuffle_word_order(self, sample=None):
        logging.info("Shuffling order of the words using texttattack.")
        logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for augmenter in self.augmenters:
                augmented = augmenter.augment(query)
                for q_variation in augmented:
                    query_variations.append([query, q_variation, augmenter.transformation.__class__.__name__, "ordering"])
            i+=1
            if sample and i > sample:
                break
        return query_variations