from transformers import pipeline
from IPython import embed
from tqdm import tqdm

import logging

class ParaphraseActions():
    def __init__(self, queries):
        self.queries = queries

    def seq2seq_paraphrase(self, sample=None):
        logging.info("Paraphrasing using pre-trained (and fine-tuned for paraphrasing) transformer seq2seq models .")
        pipelines = [
            #Fine-tuned on QQP https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
            ('ramsrigouthamg/t5_paraphraser', pipeline("text2text-generation", model = "ramsrigouthamg/t5_paraphraser", device=1)),            
            #Fine-tuned on GooglePAWS https://github.com/Vamsi995/Paraphrase-Generator
            ('Vamsi/T5_Paraphrase_Paws', pipeline("text2text-generation", model = "Vamsi/T5_Paraphrase_Paws", device=2)),
            #Fine-tuned on both https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
            ('ceshine/t5-paraphrase-quora-paws', pipeline("text2text-generation", model = 'ceshine/t5-paraphrase-quora-paws', device=2))
        ]

        logging.info("Pre-trained models used: {}.".format(str([p[0] for p in pipelines])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for pipeline_name, text2text in pipelines:
                paraphrase = text2text("paraphrase : {}? </s>".format(), num_beams=4, max_length = 20)
                query_variations.append([query, paraphrase[0]['generated_text'], pipeline_name])
            i+=1
            if sample and i > sample:
                break
        return query_variations

    # def back_translation_paraphrase(self, sample=None):
    #     logging.info("Paraphrasing using translation transformer seq2seq models (foward and back trnslation).")