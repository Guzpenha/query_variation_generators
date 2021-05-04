from textattack.augmentation import Augmenter
from textattack.transformations import *
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import (
    StopwordModification,
)

from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 

EMBEDDING_CONSTRAINT =[StopwordModification(), WordEmbeddingDistance(min_cos_sim=0.8)]
LM_CONSTRAINT = [StopwordModification(), UniversalSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )]

class SynonymActions():
    def __init__(self, queries, q_ids):
        self.queries = queries
        self.q_ids = q_ids
        self.augmenters = [
            Augmenter(transformation=WordSwapEmbedding(), transformations_per_example=1, constraints=EMBEDDING_CONSTRAINT),
            # Augmenter(transformation=WordSwapHowNet(), transformations_per_example=1),
            Augmenter(transformation=WordSwapWordNet(), transformations_per_example=1, constraints=LM_CONSTRAINT),
            Augmenter(transformation=WordSwapMaskedLM(method="bae", max_candidates=15, max_length=125), transformations_per_example=1, constraints=LM_CONSTRAINT)
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