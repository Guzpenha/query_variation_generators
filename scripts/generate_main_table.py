from IPython import embed
import pandas as pd
from os import walk
import logging
import argparse
import functools

cat = {"OriginalQuery": "original_query",
    "QueriesFromWordSwapNeighboringCharacterSwap": "misspelling",
    "QueriesFromWordSwapRandomCharacterSubstitution": "misspelling",
    "QueriesFromWordSwapQWERTY":"misspelling",
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

model_cat = {
    'BM25' : 'Trad',
    'BM25+RM3' : 'Trad', 
    'BM25+KNRM': 'NN', 
    'msmarco.convknrm.seed42.tar.gz': 'NN',
    'BM25+BERT': 'TNN',
    'BM25+T5': 'TNN',
    'msmarco.epic.seed42.tar.gz': 'TNN'
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
    task = "msmarco-passage-trec-dl"
    # task='antique'
    metric = 'ndcg_cut_10'

    _, _, filenames = next(walk(path))
    dfs = []
    for f in filenames:
        if 'query_rewriting' in f and 'per_query' not in f and  task in f:
           df = pd.read_csv(path+f)
           df["variation_method"] = df.apply(lambda r: r['name'].split("+")[-1] if 'Queries' in r['name'].split("+")[-1] else 'OriginalQuery', axis=1)
           baseline_v = df[df["variation_method"] == 'OriginalQuery'][metric].values[0]
           df[metric] = df.apply(lambda r, metric=metric, v=baseline_v: 
                round(r[metric], 4), axis=1)
                # round(r[metric], 4) if r['variation_method']=='OriginalQuery' else str(round(((r[metric]-v)/v) * 100, 2)) + "%", axis=1)
           df[metric + " p-value"] = df.apply(lambda r, metric=metric: r[metric + " p-value"] < 0.05/10,axis=1)
           model_name = f.split("model_")[-1].split(".csv")[0]
           df = df.rename(columns = {metric : "{}_".format(metric) + model_name,
                                     metric + " p-value": "p_"+ model_name})
           df = df.set_index('variation_method')
           dfs.append(df[["{}_".format(metric) + model_name, "p_"+ model_name]])
    df_all = functools.reduce(lambda df1, df2: df1.join(df2), dfs)
    df_all['category'] = df_all.apply(lambda r, m=cat: m[r.name], axis=1)    
    df_all = df_all[["category", "{}_BM25".format(metric), "p_BM25", #"{}_BM25+docT5query".format(metric), "p_BM25+docT5query",
                "{}_BM25+RM3".format(metric), "p_BM25+RM3",
                "{}_BM25+KNRM".format(metric), "p_BM25+KNRM",
                "{}_msmarco.convknrm.seed42.tar.gz".format(metric), "p_msmarco.convknrm.seed42.tar.gz",
                "{}_msmarco.epic.seed42.tar.gz".format(metric), "p_msmarco.epic.seed42.tar.gz",
                "{}_BM25+BERT".format(metric), "p_BM25+BERT",
                 "{}_BM25+T5".format(metric), "p_BM25+T5"]]
    df_all.round(4).to_csv("{}main_table_{}.csv".format(path, task), sep='\t')
    # embed()


if __name__ == "__main__":
    main()