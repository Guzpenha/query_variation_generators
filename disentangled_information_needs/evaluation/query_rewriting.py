from IPython import embed
from pyterrier_t5 import MonoT5ReRanker

import pyterrier_doc2query
import pyterrier as pt
if not pt.started():
  pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

import re
import pandas as pd
import ir_datasets
import argparse
import logging
import os
import functools
import onir_pt
import wget
import zipfile

def pair_iter(dataset):
    ds = dataset.irds_ref()
    doc_lookup = ds.docs_store()
    query_lookup = {q.query_id: q for q in ds.queries_iter()}
    while True:
        for docpair in ds.docpairs_iter():            
            text_a = doc_lookup.get(docpair.doc_id_a).text
            text_b = doc_lookup.get(docpair.doc_id_b).text
            text_query = query_lookup[docpair.query_id].text
            yield onir_pt.TrainPair(docpair.query_id, text_query,
                                    docpair.doc_id_a, text_a,
                                    docpair.doc_id_b, text_b)

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

    if args.retrieval_model_name == "BM25":
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold 
    elif args.retrieval_model_name == "BM25+BERT":
        train_ds = pt.get_dataset(args.train_dataset)
        model_path = '{}/{}_max_iter_{}_for_{}'.format(args.output_dir, args.retrieval_model_name.split("+")[1], args.max_iter, args.task.replace("/",'-'))
        if not os.path.isfile(model_path):
            logging.info("Fitting BERT.")
            vbert = onir_pt.reranker('vanilla_transformer', 'bert', vocab_config={'train': True}, config={'max_train_it':args.max_iter, 'learning_rate': 1e-5, 'batch_size': 2,  'pre_validate': False})
            if 'msmarco' in args.task:
                bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % args.cutoff_threshold >> pt.text.get_text(dataset, 'text')
                validation_run = bm25(dataset.get_topics())
                vbert.fit(va_run=validation_run, va_qrels=dataset.get_qrels(),tr_pairs=pair_iter(train_ds))
            else:
                retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(train_ds, 'text') >> vbert
                retrieval_model.fit(
                    train_ds.get_topics(),
                    train_ds.get_qrels(),
                    dataset.get_topics(),
                    dataset.get_qrels())
            vbert.to_checkpoint(model_path)
        logging.info("Loading trained BERT.")
        vbert = onir_pt.reranker.from_checkpoint(model_path)
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> vbert

    elif args.retrieval_model_name == "BM25+KNRM":
        train_ds = pt.get_dataset(args.train_dataset)
        model_path = '{}/{}_max_iter_{}_for_{}'.format(args.output_dir, args.retrieval_model_name.split("+")[1], args.max_iter, args.task.replace("/",'-'))
        if not os.path.isfile(model_path):
            logging.info("Fitting KNRM.")
            knrm = onir_pt.reranker('knrm', 'wordvec_hash', config={'max_train_it':args.max_iter, 'pre_validate': False})
            if 'msmarco' in args.task:
                bm25 = pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % args.cutoff_threshold >> pt.text.get_text(dataset, 'text')
                validation_run = bm25(dataset.get_topics())
                knrm.fit(va_run=validation_run, va_qrels=dataset.get_qrels(),tr_pairs=pair_iter(train_ds))
            else:
                retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(train_ds, 'text') >> knrm
                retrieval_model.fit(
                    train_ds.get_topics(),
                    train_ds.get_qrels(),
                    dataset.get_topics(),
                    dataset.get_qrels())
            knrm.to_checkpoint(model_path)
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
    elif 'https' in args.retrieval_model_name:
        config = {}
        if 'epic' in args.retrieval_model_name:
            config = {'batch_size': 4}
        reranker = onir_pt.reranker.from_checkpoint(args.retrieval_model_name, config=config)
        args.retrieval_model_name = args.retrieval_model_name.split("/")[-1]
        retrieval_model = pt.BatchRetrieve(index, wmodel="BM25") % args.cutoff_threshold >> pt.text.get_text(dataset, 'text') >> reranker
    # rm3_pipe = bm_25 >> pt.rewrite.RM3(index) >> bm_25
    # kl_pipe =  bm_25 >> pt.rewrite.KLQueryExpansion(index) >> bm_25

    variation_methods = []
    res_per_variation = []
    for method in query_variations['method'].unique():
        logging.info("Running model for queries generated by method {}".format(method))
        query_variation = query_variations[query_variations['method'] == method]
        query_variation['query'] = query_variation['variation']

        res_per_variation.append(retrieval_model.transform(query_variation[['qid', 'query']]))
        variation_methods.append("{}+QueriesFrom{}".format(args.retrieval_model_name, method))    

    logging.info("Calculating Kendall correlation.")
    df_scores_original_q = retrieval_model.transform(query_variations[['query','qid']].drop_duplicates())[['qid', 'docno', 'score']]
    df_scores_original_q.columns = [['qid', 'docno', 'score_original_query']]        
    score_dfs = [df_scores_original_q]
    for variation_name, df in zip(variation_methods, res_per_variation):
        df_score_only = df[['qid', 'docno', 'score']]
        df_score_only.columns = [['qid', 'docno', 'score_'+variation_name.replace("-","_").replace("/", "_").replace("+", "_")]]        
        score_dfs.append(df_score_only)

    cor_df = functools.reduce(lambda df1, df2: df1.merge(df2), score_dfs).corr(method='kendall')
    cor_df.to_csv("{}/query_corr_{}_model_{}.csv".format(args.output_dir, args.task.replace("/",'-'), args.retrieval_model_name))

    logging.info("Running remaining experiments and calculating metrics.")
    metrics = ['map', 'recip_rank', 'P_10', 'ndcg_cut_10']
    df_results = []
    df = pt.Experiment(
            # [bm_25, rm3_pipe, kl_pipe] + res_per_variation,
            [retrieval_model] + res_per_variation,
            query_variations[['query','qid']].drop_duplicates(),
            dataset.get_qrels(),
            metrics,
            baseline=0,
            names = [args.retrieval_model_name]+variation_methods)
    df.to_csv("{}/query_rewriting_{}_model_{}.csv".format(args.output_dir, args.task.replace("/",'-'), args.retrieval_model_name), index=False)

if __name__ == "__main__":
    main()