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
    path = "/home/guzpenha/personal/disentangled_information_needs/data/results/"
    # task = "msmarco-passage-trec-dl"
    task='antique'
    metric = 'ndcg_cut_10'

    _, _, filenames = next(walk(path))
    dfs = []
    for f in filenames:
        if 'query_rewriting' in f and 'per_query' not in f and task in f:
           df = pd.read_csv(path+f)
           df["variation_method"] = df.apply(lambda r: r['name'].split("+")[-1] if 'Queries' in r['name'].split("+")[-1] else 'OriginalQuery', axis=1)
           baseline_v = df[df["variation_method"] == 'OriginalQuery'][metric].values[0]
        #    df[metric] = df.apply(lambda r, metric=metric, v=baseline_v: 
        #         round(r[metric], 4) if r['variation_method']=='OriginalQuery' else str(round(((r[metric]-v)/v) * 100, 2)) + "%", axis=1)
           df[metric] = df.apply(lambda r, metric=metric, v=baseline_v: round(r[metric], 4), axis=1)
           df[metric + " p-value"] = df.apply(lambda r, metric=metric: r[metric + " p-value"] < 0.05,axis=1)
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
    df_all.reset_index().sort_values(["category", "variation_method"]).round(4).to_csv("{}main_table_{}.csv".format(path, task), sep='\t', index=False)
    

    _, _, filenames = next(walk(path))
    dfs = []
    dfs_raw = []
    df_count = []
    for f in filenames:
        if 'query_rewriting' in f and 'per_query' in f and task in f:
           df = pd.read_csv(path+f)                      
           df_count = df[df['measure']==metric].groupby(["name_x"])['qid'].count().reset_index()
           df_count["name_x"] = df_count.apply(lambda r: "QueriesFrom"+r['name_x'].split("QueriesFrom")[-1], axis=1)
           df_count.columns = ['name_x', 'count_queries']
           df['decrease_percentage'] = df['decrease_percentage'] * 100
           dfs_raw.append(df[df['measure']==metric])
           df = df[df['measure']==metric].groupby("name_x")['decrease_percentage'].mean().reset_index()
           df["name_x"] = df.apply(lambda r: "QueriesFrom"+r['name_x'].split("QueriesFrom")[-1], axis=1)
           model_name = f.split("model_")[-1].split(".csv")[0].split("_per_query")[0]
           df.columns = ['name_x', metric+"_"+model_name]
           df = df.set_index('name_x')
           dfs.append(df)
    
    dfs_raw_all = pd.concat(dfs_raw)
    dfs_raw_all["name_x"] = dfs_raw_all.apply(lambda r: "QueriesFrom"+r['name_x'].split("QueriesFrom")[-1], axis=1)
    dfs_raw_all['category'] = dfs_raw_all.apply(lambda r, m=cat: m[r['name_x']], axis=1) 
    dfs_raw_all['model_category'] = dfs_raw_all.apply(lambda r, m=model_cat: m[r['name_y']], axis=1)     
    dfs_raw_all['dataset'] = task
    dfs_raw_all.fillna(0.0).to_csv("{}per_query_all_{}.csv".format(path, task), sep='\t', index=False)
    
    df_all = functools.reduce(lambda df1, df2: df1.join(df2), dfs)
    df_all['category'] = df_all.apply(lambda r, m=cat: m[r.name], axis=1)        
    df_all = df_all[['category', '{}_BM25'.format(metric), 
                    "{}_BM25+RM3".format(metric),
                    '{}_BM25+KNRM'.format(metric), 
                    '{}_msmarco.convknrm.seed42.tar.gz'.format(metric),
                    '{}_msmarco.epic.seed42.tar.gz'.format(metric),
                    '{}_BM25+BERT'.format(metric), 
                    '{}_BM25+T5'.format(metric)]]
    df_all.merge(df_count, on='name_x').round(4).sort_values(["category", "name_x"]).\
        to_csv("{}main_table_{}_percentage_valid_queries.csv".format(path, task), sep='\t', index=False)
    # df_count.to_csv("{}main_table_{}_count.csv".format(path, task), sep='\t')    


if __name__ == "__main__":
    main()