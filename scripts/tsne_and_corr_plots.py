from IPython import embed
import pandas as pd
from sklearn.manifold import TSNE
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
    

    df = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/per_query_all_antique.csv", sep='\t')

    # task = 'antique'
    task = 'msmarco-passage-trec-dl'
    variable = 'decrease'
    for category in df['category'].unique():
        df = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/per_query_all_antique.csv", sep='\t')
        df = df[['name_x',variable,'name_y', 'dataset', 'model_category', 'qid', 'category']]
        df = df[df['category']==category]        
        pivoted_df = df[df['dataset']==task].pivot_table(variable, ['name_y', 'model_category'], ['qid', 'name_x']).reset_index()

        X = TSNE(n_components=2, perplexity=2).fit_transform(pivoted_df[pivoted_df.columns[2:]].values)
        df_to_plot = pivoted_df[pivoted_df.columns[0:2]].join(pd.DataFrame(X))
        df_to_plot.to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/tsne_{}.csv".format(category))
    
    df = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/per_query_all_antique.csv", sep='\t')
    df = df[['name_x',variable,'name_y', 'dataset', 'model_category', 'qid', 'category']]    
    pivoted_df = df[df['dataset']==task].pivot_table(variable, ['name_y', 'model_category'], ['qid', 'name_x']).reset_index()

    X = TSNE(n_components=2, perplexity=2).fit_transform(pivoted_df[pivoted_df.columns[2:]].values)
    df_to_plot = pivoted_df[pivoted_df.columns[0:2]].join(pd.DataFrame(X))
    df_to_plot['task'] = task
    df_to_plot.to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/tsne.csv".format(category))    

    custom_dict = {'': 0, 
            'BM25' : 1, 
            'BM25+RM3': 2, 
            'msmarco.convknrm.seed42.tar.gz':3, 
            'BM25+KNRM':4, 
            'msmarco.epic.seed42.tar.gz':5,
            'BM25+BERT':6, 
            'BM25+T5':7,
    } 

    pivoted_df = df[df['dataset']==task].pivot_table(variable, ['name_y', 'category'], ['qid', 'name_x']).reset_index()
    for category in pivoted_df['category'].unique():
        cat_only = pivoted_df[pivoted_df['category'] == category].dropna(axis=1)
        corr_data = cat_only[['name_y']].join(cat_only[cat_only.columns[2:]].T.corr())
        corr_data.columns = [""] + [v[0] for v in cat_only[['name_y']].values.tolist()]
        print(category)
        print(corr_data)
        
        corr_data[['', 
            'BM25', 
            'BM25+RM3', 
            'msmarco.convknrm.seed42.tar.gz', 
            'BM25+KNRM', 
            'msmarco.epic.seed42.tar.gz',
            'BM25+BERT', 
            'BM25+T5',
            ]].sort_values(by=[''], key=lambda x, c= custom_dict: x.map(c)).to_csv("/home/guzpenha/personal/disentangled_information_needs/data/results/model_corr_{}_task_{}.csv".format(category, task), sep='\t')

if __name__ == "__main__":
    main()