from IPython import embed
from pyterrier.utils import Utils
from trectools import TrecRun, TrecEval, TrecQrel, fusion
from pyterrier_t5 import MonoT5ReRanker

import pyterrier_doc2query
import pyterrier as pt
if not pt.started():
  pt.init()

import pandas as pd
import ir_datasets
import argparse
import logging
import re
import os
import sys
import numpy as np
import onir_pt
import wget
import zipfile

def combos(trec_runs, strategy="sum", max_docs=1000):
    """
        strategy: "sum", "max", "min", "anz", "mnz", "med"
        max_docs: can be either a single integer or a dict{qid,value}
    """
    dfs = []
    for t in trec_runs:
        dfs.append(t.run_data)

    if len(dfs) < 2:
        return
    merged = pd.merge(dfs[0], dfs[1], right_on =["query", "docid"] , left_on=["query", "docid"] , how = "outer", suffixes=("", "_"))
    merged = merged[["query", "q0", "docid", "score", "score_"]]

    for d in dfs[2:]:
        merged = pd.merge(merged, d, right_on=["query","docid"], left_on=["query","docid"], how="outer", suffixes=("","_"))
        merged = merged[["query", "q0", "docid", "score", "score_"]]    

    if strategy == "sum":
        merge_func = np.nansum
    elif strategy == "max":
        merge_func = np.nanmax
    elif strategy == "min":
        merge_func = np.nanmin
    elif strategy == "anz":
        merge_func = np.nanmean
    elif strategy == "mnz":
        def mnz(values):
            n_valid_entries = np.sum(~np.isnan(values))
            return np.nansum(values) * n_valid_entries
        merge_func = mnz
    elif strategy == "med":
        merge_func = np.nanmedian
    else:
        print("Unknown strategy %s. Options are: 'sum', 'max', 'min', 'anz', 'mnz'" % (strategy))
        return None

    merged["ans"] = merged[["score", "score_"]].apply(merge_func, raw=True, axis=1)
    merged.sort_values(["query", "ans"], ascending=[True,False], inplace=True)

    df = []
    for topic in merged['query'].unique():
        merged_topic = merged[merged['query'] == topic]
        if type(max_docs) == dict:
            maxd = max_docs[topic]
            for rank, entry in enumerate(merged_topic[["docid","ans"]].head(maxd).values, start=1):
                df.append([str(topic), entry[0], rank, entry[1], strategy])                
        else:
            for rank, entry in enumerate(merged_topic[["docid","ans"]].head(max_docs).values, start=1):
                df.append([str(topic), entry[0], rank, entry[1], strategy])                

    return pd.DataFrame(df, columns=["query", "docid", "rank", "score", "system"])


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
    parser.add_argument("--retrieval_model_name", default="BM25", type=str, required=False, 
                        help="[BM25, BM25+BERT, BM25+KNRM]")
    parser.add_argument("--cutoff_threshold", default=100, type=int, required=False)
    parser.add_argument("--train_dataset", default="irds:antique/train", type=str, required=False, 
                        help="train dataset for [BM25+BERT, BM25+KNRM].")
    parser.add_argument("--max_iter", default=400, type=int, required=False)
    # parser.add_argument("--ance_checkpoint", default="", type=str, required=False, 
    #                     help="ance_checkpoint from https://github.com/microsoft/ANCE/#results")                      
    args = parser.parse_args()

    query_variations = pd.read_csv(args.variations_file)
    query_variations["query"] = query_variations.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['original_query'].lower()), axis=1)
    query_variations["variation"] = query_variations.apply(lambda r, re=re: re.sub('[\W_]', ' ',  r['variation'].lower()), axis=1)
    query_variations["variation"] = query_variations.apply(lambda r: r['query'] if r['variation'].strip() == "" else r['variation'], axis=1)
    query_variations['qid'] = query_variations['q_id'].astype(str)

    dataset = pt.datasets.get_dataset(args.task)
    index_path = '{}/{}-index'.format(args.output_dir, args.task.replace('/', '-'))
    if not os.path.isdir(index_path):
        indexer = pt.index.IterDictIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter(), fields=('doc_id', 'text'))
    index = pt.IndexFactory.of(index_path+"/data.properties")
    
    if args.retrieval_model_name == "BM25":
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold 
    elif args.retrieval_model_name == "BM25+BERT":
        train_ds = pt.datasets.get_dataset(args.train_dataset)
        model_path = '{}/{}_max_iter_{}_for_{}'.format(args.output_dir, args.retrieval_model_name.split("+")[1], args.max_iter, args.task.replace("/",'-'))
        if not os.path.isfile(model_path):
            logging.info("Fitting BERT.")
            vbert = onir_pt.reranker('vanilla_transformer', 'bert', vocab_config={'train': True}, config={'max_train_it':args.max_iter, 'learning_rate': 1e-5, 'batch_size': 2})
            retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> vbert
            retrieval_model.fit(
                train_ds.get_topics(),
                train_ds.get_qrels(),
                dataset.get_topics(),
                dataset.get_qrels())
            vbert.to_checkpoint(model_path)
        else:
            logging.info("Loading trained BERT.")
            vbert = onir_pt.reranker.from_checkpoint(model_path)
            retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> vbert

    elif args.retrieval_model_name == "BM25+KNRM":
        train_ds = pt.datasets.get_dataset(args.train_dataset)
        model_path = '{}/{}_max_iter_{}_for_{}'.format(args.output_dir, args.retrieval_model_name.split("+")[1], args.max_iter, args.task.replace("/",'-'))
        if not os.path.isfile(model_path):
            logging.info("Fitting KNRM.")
            knrm = onir_pt.reranker('knrm', 'wordvec_hash', config={'max_train_it':args.max_iter})
            retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> knrm
            retrieval_model.fit(
                train_ds.get_topics(),
                train_ds.get_qrels(),
                dataset.get_topics(),
                dataset.get_qrels())
            knrm.to_checkpoint(model_path)
        else:
            logging.info("Loading trained KNRM.")
            knrm = onir_pt.reranker.from_checkpoint(model_path)
            retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> knrm
    elif args.retrieval_model_name == "BM25+T5":
        logging.info("Loading trained T5.")
        monoT5 = MonoT5ReRanker()
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> monoT5
    elif args.retrieval_model_name == "BM25+docT5query":        
        index_path_docT5query = index_path+"-docT5query"
        if not os.path.isdir(index_path_docT5query):
            if not os.path.exists("{}/t5-base.zip".format(args.output_dir)):
                wget.download("https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip", out="{}".format(args.output_dir))
                with zipfile.ZipFile('{}/t5-base.zip'.format(args.output_dir), 'r') as zip_ref:
                    zip_ref.extractall(args.output_dir)
            doc2query = pyterrier_doc2query.Doc2Query("{}/model.ckpt-1004000".format(args.output_dir), out_attr="text")
            indexer = doc2query >> pt.index.IterDictIndexer(index_path_docT5query)
            logging.info("Indexing with doc2query documents.")
            indexref = indexer.index(dataset.get_corpus_iter())
        logging.info("Loading doc2query index")
        index = pt.IndexFactory.of(index_path_docT5query+"/data.properties")
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold

    # elif args.retrieval_model_name == "ANCE":
    #     index_path_ance = '{}/{}-index-ance'.format(args.output_dir, args.task.replace('/', '-'))
    #     if not os.path.isdir(index_path_ance):
    #         indexer = pyterrier_ance.ANCEIndexer(args.ance_checkpoint, index_path_ance)
    #         indexer.index(dataset.get_corpus_iter())
    #     retrieval_model = pyterrier_ance.ANCERetrieval(args.ance_checkpoint, index_path_ance, num_results=10000)

    metrics = ['map', 'recip_rank', 'P_10', 'ndcg_cut_10']
    runs_by_type = {}
    all_runs = []

    logging.info("Running baseline model.")
    res_retrieval_baseline_model = retrieval_model.transform(query_variations[['query','qid']].drop_duplicates())
    res_retrieval_baseline_model["system"] = "{}_{}".format(args.retrieval_model_name, 'original_query')
    res_retrieval_baseline_model["query"] = res_retrieval_baseline_model['qid']
    res_retrieval_baseline_model['docid'] = res_retrieval_baseline_model['docno']
    res_retrieval_baseline_model['q0'] = ""
    res_retrieval_baseline_model.sort_values(["query","score"], inplace=True, ascending=[True,False])
    trec_run_standard_queries = TrecRun()
    trec_run_standard_queries.run_data = res_retrieval_baseline_model

    for method in query_variations['method'].unique():
        logging.info("Running model for queries generated by method {}".format(method))
        query_variation = query_variations[query_variations['method'] == method]
        query_variation['query'] = query_variation['variation']
        method_type = query_variation["transformation_type"].unique()[0]
        res = retrieval_model.transform(query_variation[['query','qid']].drop_duplicates())
        res["system"] = "{}_{}".format(args.retrieval_model_name, method)
        res["query"] = res['qid']
        res['docid'] = res['docno']
        res['q0'] = ""
        res.sort_values(["query","score"], inplace=True, ascending=[True,False])
        trec_run = TrecRun()
        trec_run.run_data = res

        if method_type not in runs_by_type:
            runs_by_type[method_type] = []
        runs_by_type[method_type].append(trec_run)
        all_runs.append(trec_run)

    logging.info("Applying rank fusion")
    fuse_methods = [
        ("CombSum", combos),
        ("RRF", fusion.reciprocal_rank_fusion)
    ]

    final_df_all = []
    for fusion_name, f in fuse_methods:
        logging.info("Fusing with {}".format(fusion_name))        
        fused_run = f(all_runs + [trec_run_standard_queries])
        if fusion_name == "RRF":
            fused_df_all = fused_run.run_data
        elif fusion_name == "CombSum":
            fused_df_all = fused_run
        fused_df_all['qid'] = fused_df_all['query']
        fused_df_all['docno'] = fused_df_all['docid']
        fused_df_all = fused_df_all[["qid", "docid", "docno", "score", "rank"]]    

        fused_by_cat = []
        for cat in runs_by_type.keys():
            logging.info("Applying rank fusion for cat {}".format(cat))
            fused_run = f(runs_by_type[cat] + [trec_run_standard_queries])
            if fusion_name == "CombSum":
                fused_df = fused_run
            else:
                fused_df = fused_run.run_data
            fused_df['qid'] = fused_df['query']
            fused_df['docno'] = fused_df['docid']
            fused_by_cat.append(fused_df)

        logging.info("Running evaluation experiment.")
        final_df = pt.Experiment(
                [res_retrieval_baseline_model, fused_df_all] + fused_by_cat,
                query_variations[['query','qid']].drop_duplicates(),
                dataset.get_qrels(),
                metrics,
                baseline=0,
                names=["{}".format(args.retrieval_model_name), 
                "{}+{}_ALL".format(args.retrieval_model_name, fusion_name)] + ["{}+{}_{}".format(args.retrieval_model_name, fusion_name, cat) for cat in runs_by_type.keys()])
        final_df_all.append(final_df)
    final_df_all = pd.concat(final_df_all).drop_duplicates()
    final_df_all.to_csv("{}/query_fusion_{}_model_{}.csv".format(args.output_dir, args.task.replace("/",'-'), args.retrieval_model_name), index=False)

if __name__ == "__main__":
    main()