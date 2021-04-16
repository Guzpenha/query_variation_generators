from IPython import embed
import pyterrier as pt
if not pt.started():
  pt.init()

import re
import pandas as pd
import ir_datasets
import argparse
import logging
import os

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
                        help="The task to evaluate (e.g. msmarco-passage/train, car/v1.5/train/fold0).")
    parser.add_argument("--variations_file", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="")
    args = parser.parse_args()

    query_variations = pd.read_csv(args.variations_file)
    query_variations["query"] = query_variations.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['original_query'].lower()), axis=1)
    query_variations["variation"] = query_variations.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['variation'].lower()), axis=1)    
    query_variations['qid'] = query_variations['q_id']

    dataset = pt.datasets.get_dataset(args.task)
    index_path = '{}/{}-index'.format(args.output_dir, args.task.replace('/', '-'))
    if not os.path.isdir(index_path):
        indexer = pt.index.IterDictIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter(), fields=('doc_id', 'text'))
    index = pt.IndexFactory.of(index_path+"/data.properties")

    tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    bm_25 = pt.BatchRetrieve(index, wmodel="BM25")

    df_results = []
    df = pt.Experiment(
            [tf_idf,bm_25],
            query_variations[['query','qid']].drop_duplicates(),
            dataset.get_qrels(),
            ['map','ndcg'])
    df['query'] = 'original_queries'
    print(df)
    df_results.append(df)    

    for method in query_variations['method'].unique():
        print(method)
        query_variation = query_variations[query_variations['method'] == method]
        query_variation['query'] = query_variation['variation']
        df = pt.Experiment(
            [tf_idf,bm_25],
            query_variation[['qid', 'query']],
            dataset.get_qrels(),
            ['map','ndcg'])
        df['query'] = 'variation_{}'.format(method)
        print(df)
        df_results.append(df)
    df_results_all = pd.concat(df_results)    
    df_results_all.to_csv("{}/query_rewriting_{}.csv".format(args.output_dir, args.task.replace("/",'-')), index=False)

if __name__ == "__main__":
    main()