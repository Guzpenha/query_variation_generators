# Query Variation Generators
This repository contains the code and annotation data for the ECIR'22 paper *"Evaluating the Robustness of Retrieval Pipelines with Query Variation Generators"*.

Instead of following the next steps you can alternatively see the same thing running on [Colab](https://colab.research.google.com/drive/1fRuTlaQnPX83o8dQHgmqs8rpUtdDue8_?usp=sharing).

One additional step is to train a T5 model using trec description data. The scripts to do this are get_trec_desc_to_q_data.py and fine_tune_summarization.py. You can also download the trained model before that [here](https://drive.google.com/file/d/1l_hzLizoZKwPAsFm3dOnezdX0SRv0LqN/view?usp=sharing).

## Setup
Install the requirements using 

    pip install -r requirements.txt

## Steps to reproduce the results
First we need to generate_weak supervsion for the desired test sets. We can do that with the scripts/generate_weak_supervision.py. In the paper we test for TREC-DL ('msmarco-passage/trec-dl-2019/judged') and ANTIQUE ('antique/train/split200-valid'), but any IR-datasets (https://ir-datasets.com/index.html) can be used here (as TASK). 

    python ${REPO_DIR}/examples/generate_weak_supervision.py 
        --task $TASK \
        --output_dir $OUT_DIR 

This will generate one query variation for each method for the original queries. After this, we manually annotated the query variations generated, in order to keep only valid ones for analysis. For that we use analyze_weak_supervision.py (prepares data for manual anotation) and analyze_auto_query_generation_labeling.py (combines auto labels and anotations.).

However, for reproducing the results we can directly use the annotated query set to test neural ranking models robustness (RQ1):

    python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py \
            --task 'irds:msmarco-passage/trec-dl-2019/judged' \
            --output_dir $OUT_DIR/ \
            --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
            --retrieval_model_name "BM25+KNRM" \
            --train_dataset "irds:msmarco-passage/train" \
            --max_iter $MAX_ITER

by using the annotated variations file directly here "\$OUT_DIR/\$VARIATIONS_FILE_TREC_DL". The same can be done to run rank fusion (RQ2) by replacing query_rewriting.py with rank_fusion.py.

The scripts [evaluate_weak_supervision.sh](https://github.com/Guzpenha/query_variation_generators/blob/main/scripts/evalueate_weak_supervision.sh) and [evaluate_rank_fusion.sh](https://github.com/Guzpenha/query_variation_generators/blob/main/scripts/evalueate_rank_fusion.sh) run all models and datasets for both research questions . The first generates the main table of results, Table 4 in the paper, and the second generates the tables for the rank fusion experiments (only available in the Arxiv version of the paper).


## Modules and Folders

- **scripts**: Contain most of the analysis scripts and also commands to run entire experiments.
- **examples**: Contain an example on how to generate query variations.
- **disentangled_information_needs/evaluation**: Scripts to evaluate robustness of models for query variations and also to evaluate rank fusion of query variations.
- **disentangled_information_needs/transformations**: Methods to generate query variations.
