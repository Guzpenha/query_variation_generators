from IPython import embed
from numpy.core.numeric import outer
import pandas as pd
from os import walk

cat = {"OriginalQuery": "original_query",
    "WordSwapNeighboringCharacterSwap": "misspelling",
    "WordSwapRandomCharacterSubstitution": "misspelling",
    "WordSwapQWERTY": "misspelling",
    "naturality_by_removing_stop_words": "naturality",
    "summarization_with_t5-base_from_description_to_title": "naturality",
    "summarization_with_t5-large": "naturality",
    "WordInnerSwapRandom": "ordering",
    "back_translation_pivot_language_de": "paraphrase",
    "ramsrigouthamg/t5_paraphraser": "paraphrase",
    "t5_uqv_paraphraser": "paraphrase",
    "WordSwapEmbedding": "paraphrase",
    "WordSwapMaskedLM": "paraphrase",
    "WordSwapWordNet": "paraphrase",
    }

def main():
    # path =  "/home/guzpenha/personal/disentangled_information_needs/data/results/per_query_all_antique.csv"
    # # path =  "/home/guzpenha/personal/disentangled_information_needs/data/results/per_query_all_msmarco-passage-trec-dl.csv"
    # df = pd.read_csv(path, sep='\t')
    # df = df[df["measure"] == 'ndcg_cut_10']
    # df.groupby(["name_x", "name_y"])['value_x'].mean()
    path = "/home/guzpenha/personal/disentangled_information_needs/data/results/oracle/"

    task='antique'
    # task='msmarco-passage-trec-dl'
    metric = 'ndcg_cut_10'
    _, _, filenames = next(walk(path))
    dfs = []    
    for f in filenames:
        if 'oracle' in f and task in f:
           df = pd.read_csv(path+f)
           df = df[df['measure']==metric]            
        #    df_count = df[df['measure']==metric].groupby(["name_x"])['qid'].count().reset_index()
        #    df_count["name_x"] = df_count.apply(lambda r: "QueriesFrom"+r['name_x'].split("QueriesFrom")[-1], axis=1)
        #    df_count.columns = ['name_x', 'count_queries']
        #    df['decrease_percentage'] = df['decrease_percentage'] * 100
        #    dfs_raw.append(df[df['measure']==metric])
        #    df = df[df['measure']==metric].groupby("name_x")['decrease_percentage'].mean().reset_index()
        #    df["name_x"] = df.apply(lambda r: "QueriesFrom"+r['name_x'].split("QueriesFrom")[-1], axis=1)
        #    model_name = f.split("model_")[-1].split(".csv")[0].split("_per_query")[0]
        #    df.columns = ['name_x', metric+"_"+model_name]
        #    df = df.set_index('name_x')
           dfs.append(df)
    dfs_raw_all = pd.concat(dfs)
    dfs_raw_all["queries"] = dfs_raw_all.apply(lambda r: r['name'].split("QueriesFrom")[-1] if r['name'].split("QueriesFrom")[-1] != r['name'] else 'original_query', axis=1)
    dfs_raw_all["model"] = dfs_raw_all.apply(lambda r: r['name'].split("+QueriesFrom")[0], axis=1)
    #Table 6 
    print("original_query")
    aux = dfs_raw_all.groupby(["model", 'queries'])['value'].mean().reset_index()    
    print(aux[aux['queries']=='original_query'][['model','value']])
    
    print("oracle")
    print(dfs_raw_all.groupby(["model", 'qid'])['value'].max().reset_index().groupby('model')['value'].mean())

    print("best query distribution")
    max_values = dfs_raw_all.groupby(["model", 'qid'])['value'].max().reset_index().merge(dfs_raw_all, on=['model', 'qid', 'value'])
    queries_with_no_gain_over_original_q = max_values[max_values['queries'] == 'original_query'][['model', 'qid']]
    queries_with_no_gain_over_original_q['remove'] = True
    max_values_filtered = max_values.merge(queries_with_no_gain_over_original_q, on=['model', 'qid'], how='outer')
    max_values_filtered = max_values_filtered[max_values_filtered['remove'] != True]
    best_query_dist = max_values_filtered.groupby(['model', 'queries'])['qid'].count().reset_index().sort_values(by=['model', 'qid'], ascending=False)
    print(best_query_dist)
    best_query_dist['category'] = best_query_dist.apply(lambda r, m=cat: m[r['queries']], axis=1)    
    best_query_dist['dataset'] = task
    best_query_dist.to_csv("{}best_query_dist.csv".format(path), index=False)
    embed()

if __name__ == "__main__":
    main()