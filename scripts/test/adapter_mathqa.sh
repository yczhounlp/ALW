export CUDA_VISIBLE_DEVICES=7

# test with best classifier
# model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'
# model='llama1-65b'
model='llama3-8b'

declare -a lr_list=(
    1e-05
    2e-05
    5e-05
    0.0001
    # 0.001
)

for lr in "${lr_list[@]}"; do
    #MathQA
    log_name=./logs/${model}/MathQA_test.log
    nohup python -u ./test.py \
        --dataset 'MathQA'  \
        --model ${model} \
        --data-path ./data/training_data/${model}/MathQA \
        --classifier /models--roberta-base \
        --classifier-pths ./ckpt/${model}/MathQA/lr-epoch-bs-${lr}-20-32 \
        --result-path ./ckpt/${model}/MathQA/lr-epoch-bs-${lr}-20-32 \
        --eval-type 'log_likelihood' \
        > ${log_name} 2>&1 &
    wait
done