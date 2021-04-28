REPO_DIR=/ssd/gustavo/disentangled_information_needs

export CUDA_VISIBLE_DEVICES=0,1
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
SAMPLE=10

# for TASK in 'antique/train/split200-valid' 'msmarco-passage/dev/small'
for TASK in 'msmarco-passage/trec-dl-2019/judged'
do
    python ${REPO_DIR}/examples/generate_weak_supervision.py --task $TASK \
        --output_dir $OUT_DIR 
        #\
        # --sample $SAMPLE
done