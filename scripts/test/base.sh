export CUDA_VISIBLE_DEVICES=5,6

# model='llama1-7b'
# model='llama1-13b'
model='llama1-30b'
# model='llama1-65b'
# model='llama3-8b'

# test without any additional operation


python -u ./test.py \
    --dataset 'ARC_C' \
    --model ${model} \
    --data-path ./data/training_data/${model}/ARC-C \
    --result-path ./results/maxprob/${model}/ARC-C \
    --eval-type 'base_log' \

# python -u ./test.py \
#     --dataset 'Gsm8k' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/Gsm8k \
#     --result-path ./results/maxprob/${model}/Gsm8k \
#     --eval-type 'base_gen' \

# python -u ./test.py \
#     --dataset 'MathQA' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/MathQA \
#     --result-path ./results/maxprob/${model}/MathQA \
#     --eval-type 'base_log' \

# python -u ./test.py \
#     --dataset 'StraQA' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/StraQA \
#     --result-path ./results/maxprob/${model}/StraQA \
#     --eval-type 'base_gen' \

# python -u ./test.py \
#     --dataset 'PiQA' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/PiQA \
#     --result-path ./results/maxprob/${model}/PiQA \
#     --eval-type 'base_log' \

# python -u ./test.py \
#     --dataset 'Folio' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/Folio \
#     --result-path ./results/maxprob/${model}/Folio \
#     --eval-type 'base_gen' \

# python -u ./test.py \
#     --dataset 'MMLU' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/MMLU \
#     --result-path ./results/maxprob/${model}/MMLU \
#     --eval-type 'base_log' \

# python -u ./test.py \
#     --dataset 'BBH' \
#     --model ${model} \
#     --data-path ./data/training_data/${model}/BBH \
#     --result-path ./results/maxprob/${model}/BBH \
#     --eval-type 'base_log' \