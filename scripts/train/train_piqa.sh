# train adapter
export CUDA_VISIBLE_DEVICES=1

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
    # PiQA
    log_name=./logs/${model}/piqa.32.20.${lr}.log
    nohup python -u ./train.py \
        --classifier '/models--roberta-base' \
        --data-path ./data/training_data/${model}/PiQA \
        --save-path ./ckpt/${model}/PiQA \
        --llm ${model} \
        --epoch 20 \
        --batch-size 32 \
        --save-every 150 \
        --print-every 150 \
        --lr ${lr} \
        > ${log_name} 2>&1 &
    wait
done