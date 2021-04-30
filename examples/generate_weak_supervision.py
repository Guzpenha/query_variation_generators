from disentangled_information_needs.transformations.synonym import SynonymActions
from disentangled_information_needs.transformations.paraphrase import ParaphraseActions
from disentangled_information_needs.transformations.naturality import NaturalityActions
from disentangled_information_needs.transformations.mispelling import MispellingActions
from disentangled_information_needs.transformations.ordering import OrderingActions
from IPython import embed

import pandas as pd
import ir_datasets
import argparse
import logging

def main():
    logging_level = logging.INFO
    logging_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging_level)
        root_handler = root_logger.handlers[0]
        root_handler.setFormatter(logging.Formatter(logging_fmt))
    except IndexError:
        logging.basicConfig(level=logging_level, format=logging_fmt)

    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="The task to generate weak supervision for (e.g. msmarco-passage/train, car/v1.5/train/fold0).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output weak supervision")
    parser.add_argument("--sample", default=None, type=int, required=False,
                        help="Number of queries to sample (if None all queries are used)")
    args = parser.parse_args()

    # queries = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/uqv100-backstories.tsv", sep="\t")
    # queries = queries["TRECTopicQuery"]
    # queries = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/uqv100-systemInputRun-uniqueOnly-spelledNormQueries.tsv", sep="\t")
    # queries = queries[queries.columns[-1]]
    logging.info("Generating weak supervision for task {} and saving results in {}."\
        .format(args.task, args.output_dir))    

    dataset = ir_datasets.load(args.task)
    queries = [t[1].lower() for t in dataset.queries_iter()]
    q_ids = [t[0] for t in dataset.queries_iter()]

    pa = ParaphraseActions(queries, q_ids, args.output_dir)
    transformed_queries_paraphrase_models = pa.seq2seq_paraphrase(sample=args.sample)
    transformed_queries_back_translation = pa.back_translation_paraphrase(sample=args.sample)
    
    na = NaturalityActions(queries, q_ids)
    transformed_queries_trec_desc_to_title = na.naturality_by_trec_desc_to_title(model_path=args.output_dir, sample=args.sample)
    transformed_queries_stop_word_removal = na.remove_stop_words(sample=args.sample)
    # transformed_queries_stop_word_and_stratified_removal = na.remove_stop_words_and_stratify_by_len(sample=args.sample)
    transformed_queries_summarizer = na.naturality_by_summarization(sample=args.sample)

    sa = SynonymActions(queries, q_ids)
    transformed_queries_syn = sa.adversarial_synonym_replacement(sample=args.sample)

    oa = OrderingActions(queries, q_ids)
    transformed_queries_shuffled_order = oa.shuffle_word_order(sample=args.sample)

    ma = MispellingActions(queries, q_ids)
    transformed_queries_mispelling = ma.mispelling_chars(sample=args.sample)


    transformed_queries =  transformed_queries_mispelling +\
      transformed_queries_shuffled_order  +\
      transformed_queries_syn + \
      transformed_queries_paraphrase_models + \
      transformed_queries_back_translation +  \
      transformed_queries_stop_word_removal +  \
      transformed_queries_summarizer + \
      transformed_queries_trec_desc_to_title

    transformed_queries = pd.DataFrame(transformed_queries, columns = ["q_id", "original_query", "variation", "method", "transformation_type"])
    transformed_queries.sort_values(by=["q_id", "transformation_type", "method"]).to_csv("{}/{}_weakly_supervised_variations_sample_{}.csv".format(args.output_dir, 
        args.task.replace("/",'-'), args.sample), index=False)

if __name__ == "__main__":
    main()