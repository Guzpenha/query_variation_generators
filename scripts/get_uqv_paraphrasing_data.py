from IPython import embed
import pandas as pd


def main():
    df = pd.read_csv("../data/uqv100-systemInputRun-uniqueOnly-spelledNormQueries.tsv", sep="\t", names=['0', 'query'])
    df['qid'] = df.apply(lambda x: x['0'].split("|")[2], axis=1)
    
    all_pairs = df.merge(df, on='qid')
    all_pairs = all_pairs[all_pairs['query_x'] != all_pairs['query_y']]
    all_pairs = all_pairs[['qid', 'query_x', 'query_y']]
    all_pairs.to_csv("../data/uqv100_pairs.csv", index=False)

if __name__ == "__main__":
    main()