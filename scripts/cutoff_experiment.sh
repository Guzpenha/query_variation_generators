export CUDA_VISIBLE_DEVICES=3
REPO_DIR=/ssd/gustavo/disentangled_information_needs
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
MAX_ITER=400
VARIATIONS_FILE_TREC_DL=variations_trec2019_labeled.csv


for T in 10 100 1000 5000
do
    python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
            --output_dir $OUT_DIR \
            --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
            --cutoff_threshold $T

    # python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
    #         --output_dir $OUT_DIR/ \
    #         --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
    #         --retrieval_model_name "BM25+RM3" \
    #         --cutoff_threshold $T

    python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
            --output_dir $OUT_DIR/ \
            --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
            --retrieval_model_name "BM25+BERT" \
            --train_dataset "irds:msmarco-passage/train" \
            --max_iter $MAX_ITER \
            --cutoff_threshold $T

    # python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
    #         --output_dir $OUT_DIR/ \
    #         --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
    #         --retrieval_model_name "BM25+KNRM" \
    #         --train_dataset "irds:msmarco-passage/train" \
    #         --max_iter $MAX_ITER \
    #         --cutoff_threshold $T

    # python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
    #         --output_dir $OUT_DIR/ \
    #         --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
    #         --retrieval_model_name "BM25+T5" \
    #         --cutoff_threshold $T

    # python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
    #         --output_dir $OUT_DIR/ \
    #         --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
    #         --retrieval_model_name "https://macavaney.us/pt_models/msmarco.epic.seed42.tar.gz" \
    #         --cutoff_threshold $T

    # python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
    #         --output_dir $OUT_DIR/ \
    #         --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
    #         --retrieval_model_name "https://macavaney.us/pt_models/msmarco.convknrm.seed42.tar.gz" \
    #         --cutoff_threshold $T
done