# train adapter
export CUDA_VISIBLE_DEVICES=5

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
    # MathQA
    log_name=./logs/${model}/mathqa.32.20.${lr}.log
    nohup python -u ./train.py \
        --classifier '/models--roberta-base' \
        --data-path ./data/training_data/${model}/MathQA \
        --save-path ./ckpt/${model}/MathQA \
        --llm ${model} \
        --epoch 20 \
        --batch-size 32 \
        --save-every 60 \
        --print-every 60 \
        --lr ${lr} \
        > ${log_name} 2>&1 &

    wait
done