from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import argparse
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import os
from collections import Counter
import numpy as np
import random
import torch.nn as nn
from data import model
import h5py
from torch.cuda.amp import autocast, GradScaler

class H5Dataset(Dataset):
    def __init__(self, args, file_path):
        self.file_path = file_path
        # 因为数据集中有-1，所以要将-1,0,1~16映射到0,1,2~17给分类器训练，和llm推理时反向映射即可
        self.label_to_int = {label: i for i, label in enumerate(range(-1, args.num_labels-1))}
        with h5py.File(file_path, 'r') as f:
            self.tensors = f['tensor'][:]
            self.texts = [text.decode('utf8') for text in f['texts'][:]]
            self.labels = f['labels'][:]

        self.labels = np.array([self.label_to_int[i] for i in self.labels])
        self.texts = np.array([i.strip() for i in self.texts])
        
        self.tensors = torch.from_numpy(self.tensors).to('cuda')
        self.labels = torch.from_numpy(self.labels).to('cuda')

# torch.from_numpy(
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.texts[idx], self.labels[idx]

def split_dataset(args):
    folder_path = os.path.join(args.data_path, 'all')
    train_path = os.path.join(args.data_path, 'train')
    valid_path = os.path.join(args.data_path, 'valid')
    test_path = os.path.join(args.data_path, 'test')

    for directory in [train_path, valid_path, test_path]:
        os.makedirs(directory, exist_ok=True)

    all_train = pd.DataFrame()
    all_valid = pd.DataFrame()
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        
        df = pd.read_csv(file_path)

        train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        all_train = all_train._append(train_data)
        all_valid = all_valid._append(val_data)
        test_data.to_csv(os.path.join(test_path, file_name), index=False)
        
    all_train.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    all_valid.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
    print(f"Train data saved in {os.path.join(train_path, 'train.csv')}")
    print('Done.')

def dataloader_h5(args, tokenizer, flag):

    h5file = os.path.join(args.data_path, 'train/data.h5')
    dataset = H5Dataset(args, h5file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    layer_counts = Counter(dataset.labels.cpu().tolist())
    sorted_counts = np.array([layer_counts[key] for key, _ in sorted(layer_counts.items())])
    weights = 1 / sorted_counts
    weights = weights / np.sum(weights)

    return dataloader, torch.from_numpy(weights).float().cuda()

def dataloader(args, tokenizer, flag):
    if args.classifier == 'head' and flag == 'train':
        return dataloader_h5(args, tokenizer, flag)

    # 因为数据集中有-1，所以要将-1,0,1~16映射到0,1,2~17给分类器训练，和llm推理时反向映射即可
    label_to_int = {label: i for i, label in enumerate(range(-1, args.num_labels-1))}

    # split train, valid, test if only 'all' folder exists
    # TODO: 把split dataset移到make training data里面去
    if not os.path.exists(os.path.join(args.data_path, 'train')):
        print('Spliting Dataset...')
        split_dataset(args)
    else:
        print('Dataset already split.')

    data_files = {}
    data_files["train"] = os.path.join(args.data_path, 'train/train.csv')
    data_files["valid"] = os.path.join(args.data_path, 'valid/valid.csv')
    raw_dataset = load_dataset('csv', data_files=data_files)

    def collate_fn(data):
        context = [i['context'].strip() for i in data]
        label = [label_to_int[i['best_layer']] for i in data]
  
        inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=context, truncation=True,
                        padding='max_length', max_length=args.max_len, return_tensors='pt').to('cuda')

        inputs['labels'] = torch.tensor(label).to('cuda')
        inputs['context'] = context
        
        return inputs
 
    dataset = raw_dataset[flag]
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    
    layer_counts = Counter(raw_dataset[flag]['best_layer'])
    # import pdb; pdb.set_trace()
    sorted_counts = np.array([layer_counts[key] for key, _ in sorted(layer_counts.items())])
    weights = 1 / sorted_counts
    weights = weights / np.sum(weights)

    return dataloader, torch.from_numpy(weights).float().cuda()

def load_classifier(args):
    label_dict = {'llama1-7b': 18, 'llama1-13b': 22, 'llama1-30b': 32, 'llama1-65b': 42, 'llama3-8b': 18}
    dim_dict = {'llama1-7b': 4096, 'llama1-13b': 5120}
    args.num_labels = label_dict[args.llm]
    
    # Load pretrain model and tokenizer
    if args.classifier == 'head':
        model = nn.Linear(dim_dict[args.llm], args.num_labels, dtype=torch.float16)
        tokenizer = None
        
    else:
        config = RobertaConfig.from_pretrained(args.classifier, num_labels=args.num_labels, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
        tokenizer = RobertaTokenizer.from_pretrained(args.classifier)
        model = RobertaForSequenceClassification.from_pretrained(args.classifier, config=config)

    return args, model.cuda(), tokenizer

def save_and_valid(args, step, classifier, ValidLoader):
    # save
    template = 'lr-epoch-bs-{:}-{:}-{:}'
    current_dict = os.path.join(args.save_path, template.format(args.lr, args.epoch, args.batch_size))

    if not os.path.exists(current_dict):
        os.makedirs(current_dict)

    torch.save(classifier.state_dict(), os.path.join(current_dict, f"{str(step)}.pth"))

    # valid
    print('\nValiding...')
    classifier.eval()
    correct = 0
    total_loss = 0
    total_num = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in ValidLoader:
            outputs = classifier(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'])

            target = batch['labels']
            
            loss = loss_fn(outputs.logits, target)
            total_loss += loss*target.shape[0]

            classify_prob = outputs.logits.softmax(dim=-1)
            pred = torch.argmax(classify_prob, dim=-1)
            
            correct += (pred == target).sum().item()
            total_num += pred.shape[0]
    
    print('\n' + '-' * 20)
    print('Valid set: ')
    print(f'step={step}')
    print('loss: %.2f' % (total_loss / total_num))
    print('acc: %.1f %%' % (100 * correct / total_num))
    print('-' * 20)

def main(args):
    args, classifier, tokenizer = load_classifier(args)

    TrainLoader, weights = dataloader(args, tokenizer, 'train')
    ValidLoader, _ = dataloader(args, tokenizer, 'valid')

    # set optimizer, scheduler and loss function
    total_steps = len(TrainLoader) * args.epoch
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, eps=1e-7)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up, num_training_steps=total_steps)

    # set loss weights
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    loss_fn = torch.nn.CrossEntropyLoss()
    # train
    classifier.train()
    print(args)
    print('start training...')
    # track average loss
    step = 0
    total_loss = 0.0
    # scaler = GradScaler()
    for epoch in range(args.epoch):
        pbar = tqdm(TrainLoader, desc=f"Epoch: {epoch+1}")

        for batch in pbar:
            optimizer.zero_grad()

            if args.classifier == 'head':
                inputs = batch[0]
                labels = batch[2]
                
                # with autocast():
                #     outputs = classifier(inputs.float())
                #     loss = loss_fn(outputs, labels)
                outputs = classifier(inputs).float()
                loss = loss_fn(outputs, labels)
                
            else:
                outputs = classifier(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'])
            
                loss = loss_fn(outputs.logits, batch['labels'])

            loss.backward()
            optimizer.step()
            scheduler.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # updata total loss
            total_loss += loss.item()
            step += 1

            if step % args.print_every == 0:
                avg_loss = total_loss / args.print_every
                pbar.set_description(f"Epoch: {epoch+1}, Step: {step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0.0

            # save & eval
            if step % args.save_every == 0:
                save_and_valid(args, step, classifier, ValidLoader)

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    
    # classifier config
    parser.add_argument('--classifier', default='roberta-base', type=str, help='the pretrain classifier model name in huggingface or local pretrain model fp')
    parser.add_argument('--llm', required=True, choices=['llama1-7b', 'llama1-13b', 'llama1-30b', 'llama1-65b', 'llama3-8b'], type=str, help='the llm that adapters trained for')
    # parser.add_argument('--num-labels', default=18, type=int, help='eg. 18 classes for llama-7b')
    parser.add_argument('--data-path', default='', type=str, help='training data and validing data')

    # train config
    parser.add_argument('--epoch', default=3, type=int, help='training epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='training batch size')
    parser.add_argument('--max-len', default=512, type=int, help='max length of sentence')
    parser.add_argument('--save-path', default='', type=str, help='save path of finetuned models')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--warm-up', default=1000, type=int, help='warm up step')
    # parser.add_argument('--sample', default=False, action='store_true', help='use oversampling and undersampling')

    # log config
    parser.add_argument('--print-every', default=100, type=int, help='print log and valid model every few steps')
    parser.add_argument('--save-every', default=100, type=int, help='save model every few steps')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)