from IPython import embed
import pandas as pd
from os import walk
import logging
import argparse
import functools

cat = {"OriginalQuery": "original_query",
    "QueriesFromWordSwapNeighboringCharacterSwap": "misspelling",
    "QueriesFromWordSwapRandomCharacterSubstitution": "misspelling",
    "QueriesFromnaturality_by_removing_stop_words": "naturality",
    "QueriesFromsummarization_with_t5-base_from_description_to_title": "naturality",
    "QueriesFromsummarization_with_t5-large": "naturality",
    "QueriesFromWordInnerSwapRandom": "ordering",
    "QueriesFromback_translation_pivot_language_de": "paraphrase",
    "QueriesFromramsrigouthamg/t5_paraphraser": "paraphrase",
    "QueriesFromt5_uqv_paraphraser": "paraphrase",
    "QueriesFromWordSwapEmbedding": "synonym",
    "QueriesFromWordSwapMaskedLM": "synonym",
    "QueriesFromWordSwapWordNet": "synonym",
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
    path = "/ssd/gustavo/disentangled_information_needs/data/"
    # task = "msmarco-passage-trec-dl"
    task='antique'
    print(task)
    _, _, filenames = next(walk(path))
    dfs = []
    for f in filenames:
        if 'unjudged_docs' in f and task in f:
           df = pd.read_csv(path+f)
           df["method"] = df.apply(lambda r: r['method'].split("+")[-1] if 'Queries' in r['method'].split("+")[-1] else 'OriginalQuery', axis=1)
           dfs.append(df)
        
    df_all = pd.concat(dfs)
    print(df_all["percentage_delta_unjudged"].mean())
    print(df_all.groupby("method")["percentage_delta_unjudged"].mean())
    # embed()


if __name__ == "__main__":
    main()