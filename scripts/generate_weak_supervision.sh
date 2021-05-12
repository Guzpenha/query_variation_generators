REPO_DIR=/ssd/gustavo/disentangled_information_needs

export CUDA_VISIBLE_DEVICES=0
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/

for TASK in 'msmarco-passage/trec-dl-2019/judged' 'antique/train/split200-valid'
do
    python ${REPO_DIR}/examples/generate_weak_supervision.py --task $TASK \
        --output_dir $OUT_DIR 
done