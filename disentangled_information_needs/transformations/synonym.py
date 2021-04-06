from textattack.augmentation import Augmenter
from textattack.transformations import *
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 

class SynonymActions():
    def __init__(self, queries):
        self.queries = queries
     
    def adversarial_synonym_replacement(self, sample=None):
        logging.info("Replacing words with synonyms using texttattack.")
        transformation_list = [
            WordSwapEmbedding(),
            WordSwapHowNet(),
            WordSwapWordNet(),
            WordSwapMaskedLM(method="bert-attack"),
            WordSwapMaskedLM(method="bae")
        ]
        logging.info("Methods used: {}.".format(str([t.__class__.__name__ for t in transformation_list])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for t in transformation_list:
                augmenter = Augmenter(transformation=t, transformations_per_example=1)
                augmented = augmenter.augment(query)
                for q_variation in augmented:
                    query_variations.append([query, q_variation, t.__class__.__name__])
            i+=1
            if sample and i > sample:
                break
        return query_variations