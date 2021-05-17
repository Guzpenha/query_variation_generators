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

    files = ["/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/UQV_annotation - Claudia.csv",
            "/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/UQV_annotation - Gustavo.csv",
            "/home/guzpenha/personal/disentangled_information_needs/data/manual_annotations/UQV_annotation - Arthur.csv"]
    dfs = [pd.read_csv(f).set_index('id')[['Generalization/Specialization', 'Aspect Change', 'Paraphrasing', 'Naturality', 'Word ordering']] for f in files]
    common_ids = functools.reduce(lambda df1, df2: df1.join(df2, rsuffix='_r', how='inner'), dfs)

    ## Calculating agreement metric
    common_ids = functools.reduce(lambda df1, df2: df1.join(df2, rsuffix='_r', how='inner'), dfs)
    common_ids.columns = ['Generalization/Specialization_0', 'Aspect Change_0', 'Paraphrasing_0', 'Naturality_0', 'Word ordering_0'] + \
                         ['Generalization/Specialization_1', 'Aspect Change_1', 'Paraphrasing_1', 'Naturality_1', 'Word ordering_1'] + \
                         ['Generalization/Specialization_2', 'Aspect Change_2', 'Paraphrasing_2', 'Naturality_2', 'Word ordering_2']        
    all_f = []
    for column in ['Generalization/Specialization', 'Aspect Change', 'Paraphrasing', 'Naturality', 'Word ordering']:
        m = []
        for i, row in common_ids.iterrows():
            count_true = 0
            count_false = 0
            for j in range(0,3):            
                if row[column+"_"+str(j)]:
                    count_true+=1
                else:
                    count_false+=1            
            m.append([count_true, count_false])
        fleiss = fleiss_kappa(np.matrix(m))
        all_f.append(fleiss)
        print("Fleiss Kappa for category {}: {}".format(column, fleiss))
    average = np.mean(all_f)
    print("Average Fleiss: {}".format(average))

    kappas = []
    for column in ['Generalization/Specialization', 'Aspect Change', 'Paraphrasing', 'Naturality', 'Word ordering']:
        for combination in [(column+"_0", column+"_1"),
                            (column+"_0", column+"_2"),
                            (column+"_1", column+"_2")]:            
            kappa = cohen_kappa_score(common_ids[combination[0]].values, common_ids[combination[1]].values)
            kappas.append(kappa)
            print("Kappa for combination {}: {}".format(combination, kappa))
    print("Average kappas : {}".format(np.mean(kappas)))

    # <0 less than chance agreement
    # 0.01-0.20 slight agreement
    # 0.21-0.40 fair agreement
    # 0.41-0.60 moderate agreement
    # 0.61-0.80 substantial agreement
    # 0.81-0.99 almost perfect agreement

    ## Count more than one category
    df = pd.concat(dfs)
    print("A total of {} combinations of query out of 600 have more than 1 category.".format(df[df.sum(axis=1)>1].shape[0]))

    #Count per category
    df_value_counts = []
    for column in ['Generalization/Specialization', 'Aspect Change', 'Paraphrasing', 'Naturality', 'Word ordering']:
        df_value_counts.append([column, df[column].value_counts().values[1]])
    df_value_counts =  pd.DataFrame(df_value_counts, columns=["Category", "count"])
    print(df_value_counts)

if __name__ == "__main__":
    main()