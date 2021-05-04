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
        ('antique', pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/antique-train-split200-valid_weakly_supervised_variations_sample_None.csv")),
        ('msmarco', pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/msmarco-passage-trec-dl-2019-judged_weakly_supervised_variations_sample_None.csv"))
    ]

    for task, df in dfs:
        logging.info(task)
        logging.info("queries: {}".format(len(df['q_id'].unique())))

        df["original_query"] = df.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['original_query'].lower().strip()), axis=1)
        df["variation"] = df.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['variation'].lower().strip()), axis=1)
        
        df['is_valid'] = " "
        df['follows_subgroup'] = " "
        df.loc[df['transformation_type'] == 'mispelling', 'is_valid'] = "yes"
        df.loc[df['transformation_type'] == 'ordering', 'is_valid'] = "yes"
        df.loc[df['method'] == 'naturality_by_removing_stop_words', 'is_valid'] = "yes"
        df.loc[df['transformation_type'] == 'mispelling', 'follows_subgroup'] = "yes"
        df.loc[df['transformation_type'] == 'ordering', 'follows_subgroup'] = "yes"
        df.loc[df['method'] == 'naturality_by_removing_stop_words', 'follows_subgroup'] = "yes"

        df.loc[df['original_query'] == df['variation'], 'is_valid'] = "no"
        df.loc[df['original_query'] == df['variation'], 'follows_subgroup'] = "no"
        df[["q_id", "method", "transformation_type", "original_query", "variation", "is_valid", "follows_subgroup"]].\
            sort_values(by=["transformation_type", "method"]).\
            to_csv("query_variations_{}.csv".format(task), sep='\t', index=False)

        df['query_equal_variation'] = df.apply(lambda r: r['original_query'] == r['variation'], axis=1)

        logging.info("Exact matches: {} out of {}.".format(df[df['query_equal_variation']].shape[0], df.shape[0]))
        print(df[df['query_equal_variation']].groupby(["method", "query_equal_variation"]).count()['q_id'])
        
        df["variation_len_change"] = df.apply(lambda r: (len(r['variation'].split(" "))-len(r['original_query'].split(" "))), axis=1)
        df["variation_len_change_%"] = df.apply(lambda r: (len(r['variation'].split(" "))-len(r['original_query'].split(" ")))/len(r['original_query'].split(" ")), axis=1)
        df["variation_len_change_%"] = df.apply(lambda r: round(r["variation_len_change_%"]* 100, 4), axis=1)
        logging.info("Variation in length: ")
        print(df.groupby(["method", 'transformation_type'])[['variation_len_change','variation_len_change_%']].mean())

if __name__ == "__main__":
    main()