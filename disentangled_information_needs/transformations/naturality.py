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
            ('t5-large', pipeline("summarization", model="t5-large", device=1)),
            ('google/pegasus-large', pipeline("summarization", model="google/pegasus-large", device=1)),
            ('facebook/bart-large-cnn', pipeline("summarization", model="facebook/bart-large-cnn", device=1))]

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
            query_variations.append([query, q_variation, "naturality_by_removing_random_words", "naturality"])
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
            query_variations.append([query, q_variation, "naturality_by_removing_stop_words", "naturality"])
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
            query_variations.append([query, q_variation, "naturality_by_removing_stop_words_and_remaining_stratified", "naturality"])
            i+=1
            if sample and i > sample:
                break
        return query_variations

    def naturality_by_summarization(self, sample=None):
        logging.info("Applying pre-trained summarization models from hugginface.")
        i=0
        batch_size=16
        query_variations = []
        for i in tqdm(range(0, len(self.queries), batch_size)):
            queries = self.queries[i:i+batch_size]
            queries_input = ["summarize : {}? </s>".format(query) for query in queries]
            for pipeline_name, summarizer in self.summarization_pipelines:
                summaries = summarizer(queries_input, min_length=3, max_length=8)
                for j, summary in enumerate(summaries):
                    query_variations.append([queries[j], summary['summary_text'], "summarization_with_{}".format(pipeline_name), "naturality"])
            i+=batch_size
            if sample and i > sample:
                break        
        return query_variations

    def naturality_by_trec_desc_to_title(self, model_path, sample=None):
        logging.info("Applying pre-trained summarization model fine-tuned for trec desc to title.")
        summarizer = pipeline("summarization", model="{}/t5-base_from_description_to_title/".format(model_path), tokenizer="t5-base")
        i=0
        batch_size=16
        query_variations = []
        for i in tqdm(range(0, len(self.queries), batch_size)):
            queries = self.queries[i:i+batch_size]
            queries_input = ["summarize : {}? </s>".format(query) for query in queries]
            summaries = summarizer(queries_input, min_length=3, max_length=8)
            for j, summary in enumerate(summaries):
                query_variations.append([queries[j], summary['summary_text'], "summarization_with_{}".format('t5-base_from_description_to_title'), "naturality"])
            i+=batch_size
            if sample and i > sample:
                break        
        return query_variations