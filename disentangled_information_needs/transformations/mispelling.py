from textattack.augmentation import Augmenter
from textattack.transformations import *
from textattack.constraints.pre_transformation import (
    StopwordModification,
)
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 


CONSTRAINTS = [StopwordModification()]

class MispellingActions():
    def __init__(self, queries, q_ids):
        self.queries = queries
        self.q_ids = q_ids
        self.augmenters = [
            Augmenter(transformation=WordSwapNeighboringCharacterSwap(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=WordSwapRandomCharacterSubstitution(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=WordSwapQWERTY(), transformations_per_example=1, constraints=CONSTRAINTS),
        ]

    def mispelling_chars(self, sample=None):
        logging.info("Adding mispelling errors using texttattack.")
        logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for augmenter in self.augmenters:
                try:
                    augmented = augmenter.augment(query)
                except: #empty error for QWERTY.
                    augmented = [query]
                for q_variation in augmented:
                    query_variations.append([self.q_ids[i], query, q_variation, augmenter.transformation.__class__.__name__, "mispelling"])
            i+=1
            if sample and i > sample:
                break
        return query_variations
