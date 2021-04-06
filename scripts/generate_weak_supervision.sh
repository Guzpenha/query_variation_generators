REPO_DIR=/ssd/gustavo/disentangled_information_needs

export CUDA_VISIBLE_DEVICES=0,1
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
SAMPLE=10

for TASK in 'msmarco-passage/train' 'car/v1.5/train/fold0' 'antique/train'
do
    python ${REPO_DIR}/examples/generate_weak_supervision.py --task $TASK \
        --output_dir $OUT_DIR \
        --sample $SAMPLE
done