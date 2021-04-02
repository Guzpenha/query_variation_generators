from disentangled_information_needs.transformations.synonym import *

import pandas as pd
import ir_datasets
import argparse
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

    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="The task to generate weak supervision for (e.g. msmarco-passage/train, car/v1.5/train/fold0).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output weak supervision")
    parser.add_argument("--sample", default=None, type=int, required=True,
                        help="Number of queries to sample (if None all queries are used)")
    args = parser.parse_args()    

    # queries = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/uqv100-backstories.tsv", sep="\t")
    # queries = queries["TRECTopicQuery"]
    # queries = pd.read_csv("/home/guzpenha/personal/disentangled_information_needs/data/uqv100-systemInputRun-uniqueOnly-spelledNormQueries.tsv", sep="\t")
    # queries = queries[queries.columns[-1]]
    logging.info("Generating weak supervision for task {} and saving results in {}."\
        .format(args.task, args.output_dir))    

    dataset = ir_datasets.load(args.task)
    queries = [t[1] for t in dataset.queries_iter()]

    sa = SynonymActions(queries)
    transformed_queries = sa.adversarial_paraphrase(sample=args.sample)
    transformed_queries = pd.DataFrame(transformed_queries, columns = ["original_query", "variation", "method"])
    transformed_queries.to_csv("{}/{}_weak_supervised_variations_synonym.csv".format(args.output_dir, args.task.split("/")[0]), index=False)

if __name__ == "__main__":
    main()