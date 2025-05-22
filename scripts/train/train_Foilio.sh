# train adapter
export CUDA_VISIBLE_DEVICES=4

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
    # Folio
    log_name=./logs/${model}/folio.32.3.${lr}.log
    nohup python -u ./train.py \
        --classifier 'models--roberta-base' \
        --data-path ./data/training_data/${model}/Folio \
        --save-path ./ckpt/${model}/Folio \
        --llm ${model} \
        --epoch 3 \
        --batch-size 32 \
        --save-every 160 \
        --print-every 160 \
        --lr ${lr} \
        > ${log_name} 2>&1 &

    wait
done