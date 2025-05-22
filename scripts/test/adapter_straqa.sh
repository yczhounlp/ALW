export CUDA_VISIBLE_DEVICES=6

# test with best classifier
# model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'
# model='llama1-65b'
model='llama3-8b'

declare -a lr_list=(
    # 1e-05
    2e-05
    5e-05
    0.0001
    # 0.001
)

for lr in "${lr_list[@]}"; do
    # Straqa
    log_name=./logs/${model}/StraQA_test.log
    nohup python -u ./test.py \
        --dataset 'StraQA' \
        --model ${model} \
        --data-path ./data/training_data/${model}/StraQA \
        --classifier /models--roberta-base \
        --classifier-pths ./ckpt/${model}/StraQA/lr-epoch-bs-${lr}-3-32 \
        --result-path ./ckpt/${model}/StraQA/lr-epoch-bs-${lr}-3-32 \
        --eval-type 'generate' \
        > ${log_name} 2>&1 &
    wait
done