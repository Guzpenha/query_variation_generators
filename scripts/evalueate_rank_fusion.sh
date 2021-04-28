export CUDA_VISIBLE_DEVICES=1
REPO_DIR=/ssd/gustavo/disentangled_information_needs
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
MAX_ITER=200

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/antique-train-split200-valid_weakly_supervised_variations_sample_None.csv

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/antique-train-split200-valid_weakly_supervised_variations_sample_None.csv \
        --retrieval_model_name "BM25+BERT" \
        --train_dataset "irds:antique/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/antique-train-split200-valid_weakly_supervised_variations_sample_None.csv \
        --retrieval_model_name "BM25+KNRM" \
        --train_dataset "irds:antique/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/msmarco-passage-trec-dl-2019-judged_weakly_supervised_variations_sample_None.csv

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/msmarco-passage-trec-dl-2019-judged_weakly_supervised_variations_sample_None.csv \
        --retrieval_model_name "BM25+BERT" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/msmarco-passage-trec-dl-2019-judged_weakly_supervised_variations_sample_None.csv \
        --retrieval_model_name "BM25+KNRM" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER



# python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/dev/small' \
#         --output_dir $OUT_DIR/ \
#         --variations_file $OUT_DIR/msmarco-passage-dev-small_weakly_supervised_variations_sample_None.csv

# python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/dev/small' \
#         --output_dir $OUT_DIR/ \
#         --variations_file $OUT_DIR/msmarco-passage-dev-small_weakly_supervised_variations_sample_None.csv \
#         -retrieval_model_name "BM25+BERT" \
#         --train_dataset "irds:msmarco-passage/train" \
#         --max_iter 100

# python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/dev/small' \
#         --output_dir $OUT_DIR/ \
#         --variations_file $OUT_DIR/msmarco-passage-dev-small_weakly_supervised_variations_sample_None.csv \
#         --retrieval_model_name "BM25+KNRM" \
#         --train_dataset "irds:msmarco-passage/train" \
#         --max_iter 100
