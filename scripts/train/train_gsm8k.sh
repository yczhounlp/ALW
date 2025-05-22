# train adapter
export CUDA_VISIBLE_DEVICES=7

# model='llama1-7b'
# model='llama1-13b'
# model='llama1-30b'
# model='llama1-65b'
model='llama3-8b'

# 越大的模型选的lr越少
declare -a lr_list=(
    1e-05
    2e-05
    5e-05
    0.0001
    # 0.001
    # 0.05
)

for lr in "${lr_list[@]}"; do
    # gsm8k
    log_name=./logs/${model}/gsm8k.32.3.${lr}.log
    nohup python -u ./train.py \
        --classifier '/models--roberta-base' \
        --data-path ./data/training_data/${model}/Gsm8k \
        --save-path ./ckpt/${model}/Gsm8k \
        --llm ${model} \
        --epoch 3 \
        --batch-size 32 \
        --lr ${lr} \
        > ${log_name} 2>&1 &
    wait
done