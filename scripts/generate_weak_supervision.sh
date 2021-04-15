REPO_DIR=/ssd/gustavo/disentangled_information_needs

export CUDA_VISIBLE_DEVICES=0,1,2
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
SAMPLE=10

for TASK in 'msmarco-passage/dev/small' 'car/v1.5/test200' 'antique/test'
do
    python ${REPO_DIR}/examples/generate_weak_supervision.py --task $TASK \
        --output_dir $OUT_DIR 
        #\
        # --sample $SAMPLE
done