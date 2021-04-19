from textattack.augmentation import Augmenter
from textattack.transformations import *
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 

class SynonymActions():
    def __init__(self, queries, q_ids):
        self.queries = queries
        self.q_ids = q_ids
        self.augmenters = [
            Augmenter(transformation=WordSwapEmbedding(), transformations_per_example=1),
            Augmenter(transformation=WordSwapHowNet(), transformations_per_example=1),
            Augmenter(transformation=WordSwapWordNet(), transformations_per_example=1),
            Augmenter(transformation=WordSwapMaskedLM(method="bae", max_candidates=1, max_length=125), transformations_per_example=1)
        ]

    def adversarial_synonym_replacement(self, sample=None):
        logging.info("Replacing words with synonyms using texttattack.")
        logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for augmenter in self.augmenters:
                augmented = augmenter.augment(query)
                for q_variation in augmented:
                    query_variations.append([self.q_ids[i], query, q_variation, augmenter.transformation.__class__.__name__, "synonym"])
            i+=1
            if sample and i > sample:
                break
        return query_variations