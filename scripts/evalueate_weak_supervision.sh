REPO_DIR=/ssd/gustavo/disentangled_information_needs
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/

python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:msmarco-passage/dev/small' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/msmarco-passage-dev-small_weakly_supervised_variations_sample_None.csv

python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:car/v1.5/test200' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/car-v1.5-test200_weakly_supervised_variations_sample_None.csv

python ${REPO_DIR}/disentangled_information_needs/evaluation/query_rewriting.py --task 'irds:antique/test' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/antique-test_weakly_supervised_variations_sample_None.csv