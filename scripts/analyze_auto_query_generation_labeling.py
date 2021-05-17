from IPython import embed
import pandas as pd
from os import walk
import logging
import argparse
import functools
import re
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score

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

    files_labeled = ["/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/automatic_query_variations_anotation - Arthur.csv",
            "/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/automatic_query_variations_anotation - Gustavo.csv",
            "/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/automatic_query_variations_anotation - Claudia.csv"]

    files_auto_labeled = ["/home/guzpenha/personal/disentangled_information_needs/data/results/query_variations_antique_labeled_auto.csv",
            "/home/guzpenha/personal/disentangled_information_needs/data/results/query_variations_msmarco_labeled_auto.csv"]

    df_auto = [pd.read_csv(f, sep="\t") for f in files_auto_labeled]
    df_labels = [pd.read_csv(f) for f in files_labeled]

    df_all = pd.concat(df_auto+df_labels)
    df_all.loc[df_all["dataset"] == "trec-dl-2019", 'dataset'] = "msmarco"

    # valid per method
    df_all.groupby(["dataset", "method", "valid"]).count()['q_id'].reset_index().\
        sort_values(["dataset", "method", "q_id"]).to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results_plot/valid_per_method.csv")

    df_percent = df_all.groupby(["dataset", "method", "valid"]).count()['q_id'].reset_index()
    df_percent['percentage'] = df_percent.apply(lambda r: (r['q_id']/200) * 100 if r['dataset'] == 'antique' else (r['q_id']/43) * 100 , axis=1)
    df_percent[df_percent["valid"]].to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results_plot/valid_per_method_percentage.csv", sep="\t", index=False)
    
    #follows subgroup per method
    df_all.groupby(["dataset", "method", "follow_category"]).count()['q_id'].reset_index().to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results_plot/follows_category_per_method.csv")

    methods_to_remove = ["t5_uqv_paraphraser", "summarization_with_t5-base"]
    df_all[(~df_all["method"].isin(methods_to_remove)) & (df_all["dataset"] == 'msmarco')].\
        to_csv("/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/variations_trec2019_labeled.csv", index=False)
    df_all[(~df_all["method"].isin(methods_to_remove)) & (df_all["dataset"] == 'antique')].fillna(False).\
        to_csv("/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/variations_antique_labeled.csv", index=False)
    embed()

if __name__ == "__main__":
    main()