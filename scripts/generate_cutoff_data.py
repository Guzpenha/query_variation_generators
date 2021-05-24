from IPython import embed
import pandas as pd
from os import walk
import logging
import argparse
import functools

cat = {"OriginalQuery": "original_query",
    "QueriesFromWordSwapNeighboringCharacterSwap": "misspelling",
    "QueriesFromWordSwapRandomCharacterSubstitution": "misspelling",
    "QueriesFromWordSwapQWERTY": "misspelling",
    "QueriesFromnaturality_by_removing_stop_words": "naturality",
    "QueriesFromsummarization_with_t5-base_from_description_to_title": "naturality",
    "QueriesFromsummarization_with_t5-large": "naturality",
    "QueriesFromWordInnerSwapRandom": "ordering",
    "QueriesFromback_translation_pivot_language_de": "paraphrase",
    "QueriesFromramsrigouthamg/t5_paraphraser": "paraphrase",
    "QueriesFromt5_uqv_paraphraser": "paraphrase",
    "QueriesFromWordSwapEmbedding": "paraphrase",
    "QueriesFromWordSwapMaskedLM": "paraphrase",
    "QueriesFromWordSwapWordNet": "paraphrase",
    }

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
    # parser.add_argument("--task", default=None, type=str, required=True,
    #                     help="The task to generate weak supervision for (e.g. msmarco-passage/train, car/v1.5/train/fold0).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="the folder to output weak supervision")
    # parser.add_argument("--sample", default=None, type=int, required=False,
    #                     help="Number of queries to sample (if None all queries are used)")
    # args = parser.parse_args()
    path = "/home/guzpenha/personal/disentangled_information_needs/data/results/cutoff/"

    _, _, filenames = next(walk(path))
    dfs = []
    for f in filenames:
        if 'BERT_per_query_cutoff_' in f:
           df = pd.read_csv(path+f)
           df["variation_method"] = df.apply(lambda r: r['name_x'].split("+")[-1] if 'Queries' in r['name_x'].split("+")[-1] else 'OriginalQuery', axis=1)                      
           df['threshold'] = f.split("cutoff_")[-1].split(".csv")[0]
           dfs.append(df)
    df_all = pd.concat(dfs)
    df_all['category'] = df_all.apply(lambda r, m=cat: m[r['variation_method']], axis=1)
    df_all.to_csv("{}cutoff_table.csv".format(path), sep='\t', index=False)    

if __name__ == "__main__":
    main()