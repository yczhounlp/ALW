# train adapter
export CUDA_VISIBLE_DEVICES=2

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
    # BBH
    log_name=./logs/${model}/bbh.32.20.${lr}.log
    nohup python -u ./train.py \
        --classifier 'models--roberta-base' \
        --data-path ./data/training_data/${model}/BBH \
        --save-path ./ckpt/${model}/BBH \
        --llm ${model} \
        --epoch 20 \
        --batch-size 32 \
        --lr ${lr} \
        --warm-up 300 \
        --save-every 30 \
        --print-every 30 \
        > ${log_name} 2>&1 &
    wait
done