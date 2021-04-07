from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from IPython import embed
from tqdm import tqdm

import logging
import torch

class ParaphraseActions():
    def __init__(self, queries):
        self.queries = queries
        self.paraphrase_pipelines = [
            #Fine-tuned on QQP https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
            ('ramsrigouthamg/t5_paraphraser', pipeline("text2text-generation", model = "ramsrigouthamg/t5_paraphraser", device=0)), 
            #Fine-tuned on GooglePAWS https://github.com/Vamsi995/Paraphrase-Generator
            ('Vamsi/T5_Paraphrase_Paws', pipeline("text2text-generation", model = "Vamsi/T5_Paraphrase_Paws", device=0)),
            #Fine-tuned on both https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
            ('ceshine/t5-paraphrase-quora-paws', pipeline("text2text-generation", model = 'ceshine/t5-paraphrase-quora-paws', device=0))
        ]

        self.pivot_languages = [
            'es', 
            'fr',
            'de'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("device used for translation models: {}".format(self.device))
        self.translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.translation_model.to(self.device)

    def seq2seq_paraphrase(self, sample=None):
        logging.info("Paraphrasing using pre-trained (and fine-tuned for paraphrasing) transformer seq2seq models .")
        logging.info("Pre-trained models used: {}.".format(str([p[0] for p in self.paraphrase_pipelines])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for pipeline_name, text2text in self.paraphrase_pipelines:
                paraphrase = text2text("paraphrase : {}? </s>".format(query), num_beams=4, max_length = 20)
                query_variations.append([query, paraphrase[0]['generated_text'], pipeline_name])
            i+=1
            if sample and i > sample:
                break
        return query_variations


    def back_translation(self, query, pivot_language='es'):
        # translate English to pivot language
        self.translation_tokenizer.src_lang = "en"
        encoded_en = self.translation_tokenizer(query, return_tensors="pt")
        encoded_en.to(self.device)
        generated_tokens = self.translation_model.generate(**encoded_en, forced_bos_token_id=self.translation_tokenizer.get_lang_id(pivot_language))
        query_pivot_language = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)            

        # translate from pivot language to English
        self.translation_tokenizer.src_lang = pivot_language
        encoded_pivot_language = self.translation_tokenizer(query_pivot_language, return_tensors="pt")
        encoded_pivot_language.to(self.device)
        generated_tokens = self.translation_model.generate(**encoded_pivot_language, forced_bos_token_id=self.translation_tokenizer.get_lang_id("en"))
        rewritten_query = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return rewritten_query[0]

    def back_translation_paraphrase(self, sample=None):
        logging.info("Paraphrasing using translation model (foward and back trnslation).")
        logging.info("Pivot languages used: {}.".format(str([p for p in self.pivot_languages])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for pivot_language in self.pivot_languages:
                paraphrase = self.back_translation(query, pivot_language=pivot_language)
                query_variations.append([query, paraphrase, 'back_translation_pivot_language_{}'.format(pivot_language)])
            i+=1
            if sample and i > sample:
                break
        return query_variations