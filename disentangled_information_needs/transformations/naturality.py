from transformers import pipeline
from textattack.augmentation.recipes import DeletionAugmenter
from textattack.transformations import WordDeletion
from textattack.augmentation import Augmenter

from IPython import embed
from tqdm import tqdm

import nltk
import logging
import random

random.seed(42)

class NaturalityActions():
    def __init__(self, queries):
        self.queries = queries
        self.summarization_pipelines = [            
            ('distilbart-cnn-12-6', pipeline("summarization", model="distilbart-cnn-12-6")),
            ('t5-large', pipeline("summarization", model="t5-large"))
            ]        

    def remove_words(self, query, pct_to_remove=0.2):
        query_splitted = query.split()
        n_terms = len(query_splitted)
        n_to_remove = max(int(pct_to_remove*n_terms), 1)

        idx_to_remove = set(random.sample(range(0, len(query_splitted)),
                n_to_remove))
        new_q = []
        for i in range(n_terms):
            if i not in idx_to_remove:
                new_q.append(query_splitted[i])
        return " ".join(new_q)

    def remove_random_words(self, sample=None):
        logging.info("Removing words randomly.")
        i=0
        query_variations = []
        for query in tqdm(self.queries):            
            q_variation = self.remove_words(query)
            query_variations.append([query, q_variation, "naturality_by_removing_random_words"])
            i+=1
            if sample and i > sample:
                break
        return query_variations

    def remove_stop_words(self, sample=None):
        logging.info("Removing stop words.")
        i=0
        stopwords = set(nltk.corpus.stopwords.words("english"))
        query_variations = []
        for query in tqdm(self.queries):            
            q_variation = " ".join([w for w in query.lower().split() if w not in stopwords])
            query_variations.append([query, q_variation, "naturality_by_removing_stop_words"])
            i+=1
            if sample and i > sample:
                break
        return query_variations

    def remove_stop_words_and_stratify_by_len(self, sample=None):
        logging.info("Removing stop words and after that remove remaining words randomly.")
        i=0
        stopwords = set(nltk.corpus.stopwords.words("english"))
        query_variations = []
        for query in tqdm(self.queries):            
            q_no_stop_words = [w for w in query.lower().split() if w not in stopwords]
            if len(q_no_stop_words) > 3: 
                q_variation = self.remove_words(" ".join(q_no_stop_words), random.randint(0, 65)/100)
            else:
                q_variation = " ".join(q_no_stop_words)
            query_variations.append([query, q_variation, "naturality_by_removing_stop_words_and_remaining_stratified"])
            i+=1
            if sample and i > sample:
                break
        return query_variations

    def naturality_by_summarization(self, sample=None):
        logging.info("Applying pre-trained summarization models from hugginface.")
        i=0
        query_variations = []
        for query in tqdm(self.queries):                        
            for pipeline_name, summarizer in self.summarization_pipelines:
                summary = summarizer(query, min_length=3, max_length=max(6, int(len(query.split()) *0.5)))
                query_variations.append([query, summary[0]['summary_text'], "summarization_with_{}".format(pipeline_name)])
            i+=1
            if sample and i > sample:
                break
        return query_variations