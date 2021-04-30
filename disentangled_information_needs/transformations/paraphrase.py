from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from IPython import embed
from tqdm import tqdm

import logging
import torch

CUDA_DEVICE = 0

class ParaphraseActions():
    def __init__(self, queries, q_ids, uqv_model_path):
        self.queries = queries
        self.q_ids = q_ids
        self.paraphrase_pipelines = [
            #Fine-tuned on QQP https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555
            ('ramsrigouthamg/t5_paraphraser', pipeline("text2text-generation", model = "ramsrigouthamg/t5_paraphraser", device=CUDA_DEVICE))
            # Fine-tuned on GooglePAWS https://github.com/Vamsi995/Paraphrase-Generator
            # ('Vamsi/T5_Paraphrase_Paws', pipeline("text2text-generation", model = "Vamsi/T5_Paraphrase_Paws", device=1)),
            #Fine-tuned on both https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
            # ('ceshine/t5-paraphrase-quora-paws', pipeline("text2text-generation", model = 'ceshine/t5-paraphrase-quora-paws', device=1))
            ('t5_uqv_paraphraser', pipeline("text2text-generation", model="{}/t5-base_uqv_variation_paraphraser/".format(uqv_model_path), tokenizer="t5-base", device=CUDA_DEVICE))
        ]

        self.pivot_languages = [
            # 'es', 
            # 'fr',
            'de'
        ]
        self.device = torch.device('cuda:{}'.format(CUDA_DEVICE) if torch.cuda.is_available() else 'cpu')
        logging.info("device used for translation models: {}".format(self.device))
        self.translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.translation_model.to(self.device)

    def seq2seq_paraphrase(self, sample=None):
        logging.info("Paraphrasing using pre-trained (and fine-tuned for paraphrasing) transformer seq2seq models .")
        logging.info("Pre-trained models used: {}.".format(str([p[0] for p in self.paraphrase_pipelines])))
        i=0
        batch_size=16
        query_variations = []        
        for i in tqdm(range(0, len(self.queries), batch_size)):
            queries = self.queries[i:i+batch_size]
            q_ids_bactch = self.q_ids[i:i+batch_size]
            queries_input = ["paraphrase : {}? </s>".format(query) for query in queries]
            for pipeline_name, text2text in self.paraphrase_pipelines:
                paraphrases = text2text(queries_input, num_beams=4, max_length = 40)
                for j, paraphrase in enumerate(paraphrases):
                    query_variations.append([q_ids_bactch[j], queries[j], paraphrase['generated_text'], pipeline_name, "paraphrase"])
            i+=batch_size
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
                query_variations.append([self.q_ids[i], query, paraphrase, 'back_translation_pivot_language_{}'.format(pivot_language), "paraphrase"])
            i+=1
            if sample and i > sample:
                break
        return query_variations