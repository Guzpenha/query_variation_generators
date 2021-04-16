from IPython import embed
from pyterrier.utils import Utils
from trectools import TrecRun, TrecEval, TrecQrel, fusion

import pyterrier as pt
if not pt.started():
  pt.init()

import pandas as pd
import ir_datasets
import argparse
import logging
import re
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
    query_variations["variation"] = query_variations.apply(lambda r: r['query'] if r['variation'].strip() == "" else r['variation'], axis=1)
    query_variations['qid'] = query_variations['q_id']

    dataset = pt.datasets.get_dataset(args.task)
    index_path = '{}/{}-index'.format(args.output_dir, args.task.replace('/', '-'))
    if not os.path.isdir(index_path):
        indexer = pt.index.IterDictIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter(), fields=('doc_id', 'text'))
    index = pt.IndexFactory.of(index_path+"/data.properties")

    bm_25 = pt.BatchRetrieve(index, wmodel="BM25")
        
    df = pt.Experiment(
            [bm_25],
            query_variations[['query','qid']].drop_duplicates(),
            dataset.get_qrels(),
            ['map'])
    print(df)

    runs = []
    for method in query_variations['method'].unique():
        query_variation = query_variations[query_variations['method'] == method]
        query_variation['query'] = query_variation['variation']

        res = bm_25.transform(query_variations[['query','qid']].drop_duplicates())        
        res["system"] = "bm25_{}".format(method)
        res["query"] = res['qid']
        res['docid'] = res['docno']
        res.sort_values(["query","score"], inplace=True, ascending=[True,False])
        trec_run = TrecRun()
        trec_run.run_data = res
        runs.append(trec_run)
    
    fused_run = fusion.reciprocal_rank_fusion(runs)
    fused_df = fused_run.run_data
    fused_df['qid'] = fused_df['query']
    fused_df['docno'] = fused_df['docid']
    eval_map = Utils.evaluate(fused_df, dataset.get_qrels(), metrics=['map'])['map']

    final_df = pd.concat([df, pd.DataFrame([["rank_fusion_all", eval_map]], columns = ['name', 'map'])])
    final_df.to_csv("{}/query_fusion_{}.csv".format(args.output_dir, args.task.replace("/",'-')), index=False)

if __name__ == "__main__":
    main()