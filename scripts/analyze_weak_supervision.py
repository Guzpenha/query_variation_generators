from IPython import embed
import pandas as pd
from os import walk
import logging
import argparse
import functools
import re

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

    dfs = [
        ('antique', pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/antique-train-split200-valid_weakly_supervised_variations_sample_None.csv")),
        ('msmarco', pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/msmarco-passage-trec-dl-2019-judged_weakly_supervised_variations_sample_None.csv"))
    ]

    for task, df in dfs:
        logging.info(task)
        logging.info("queries: {}".format(len(df['q_id'].unique())))

        df["original_query"] = df.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['original_query'].lower().strip()).strip(), axis=1)
        df["variation"] = df.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['variation'].lower().strip()).strip(), axis=1)        
        df = df[df["method"]!="WordSwapMaskedLM"]

        df.loc[df['transformation_type'] == 'mispelling', 'valid'] = "TRUE"
        df.loc[df['transformation_type'] == 'ordering', 'valid'] = "TRUE"
        df.loc[df['method'] == 'naturality_by_removing_stop_words', 'valid'] = "TRUE"
        df.loc[df['transformation_type'] == 'mispelling', 'follow_category'] = "TRUE"
        df.loc[df['transformation_type'] == 'ordering', 'follow_category'] = "TRUE"
        df.loc[df['method'] == 'naturality_by_removing_stop_words', 'follow_category'] = "TRUE"

        df.loc[df['original_query'] == df['variation'], 'valid'] = "FALSE"
        df.loc[df['original_query'] == df['variation'], 'follow_category'] = "FALSE"
        df["dataset"] = task
        df[["dataset", "q_id", "method", "transformation_type", "original_query", "variation", "valid", "follow_category"]][df["valid"].isnull()].\
            sort_values(by=["transformation_type", "method"]).\
            to_csv("../data/results/query_variations_{}_to_label.csv".format(task), sep='\t', index=False)

        df[["dataset", "q_id", "method", "transformation_type", "original_query", "variation", "valid", "follow_category"]][~df["valid"].isnull()].\
            sort_values(by=["transformation_type", "method"]).\
            to_csv("../data/results/query_variations_{}_labeled_auto.csv".format(task), sep='\t', index=False)

        df['query_equal_variation'] = df.apply(lambda r: r['original_query'] == r['variation'], axis=1)

        logging.info("Exact matches: {} out of {}.".format(df[df['query_equal_variation']].shape[0], df.shape[0]))
        print(df.groupby(["method"]).count()['q_id'])
        print(df[df['query_equal_variation']].groupby(["method", "query_equal_variation"]).count()['q_id'])
        
        df["variation_len_change"] = df.apply(lambda r: (len(r['variation'].split(" "))-len(r['original_query'].split(" "))), axis=1)
        df["variation_len_change_%"] = df.apply(lambda r: (len(r['variation'].split(" "))-len(r['original_query'].split(" ")))/len(r['original_query'].split(" ")), axis=1)
        df["variation_len_change_%"] = df.apply(lambda r: round(r["variation_len_change_%"]* 100, 4), axis=1)
        logging.info("Variation in length: ")
        print(df.groupby(["method", 'transformation_type'])[['variation_len_change','variation_len_change_%']].mean())

if __name__ == "__main__":
    main()