export CUDA_VISIBLE_DEVICES=5

root_path='./ALW_emb'

cd $root_path

# model='llama1-7b'
# layers=16
model='llama3-8b'
layers=16
# model='llama1-13b'
# layers=20
# model='llama1-30b'
# layers=30
# model='llama1-65b'
# layers=40


# python ./data/main.py \
#     --dataset 'MMLU' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'BBH' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'Gsm8k' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'StraQA' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'MathQA' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'PiQA' \
#     --model ${model} \
#     --layers ${layers} \

# python ./data/main.py \
#     --dataset 'Folio' \
#     --model ${model} \
#     --layers ${layers} \

python ./data/main.py \
    --dataset 'ARC-C' \
    --model ${model} \
    --layers ${layers} \