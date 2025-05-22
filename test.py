from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
import torch
import argparse
from data import model
import test_loader
import pandas as pd
import tqdm
import os
import glob
import torch.nn as nn
import eval
# from sentence_transformers import SentenceTransformer, models, util
import csv

def get_last_hidden(args, input_text, Model):
    input_ids = Model.tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    
    with torch.no_grad():
        outputs = Model.model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
    
        logits = outputs.hidden_states
        mature_layer = len(logits)-1

    return logits[mature_layer][0][-1]

class classifier:
    def __init__(self, args, pth=''):
        self.tokenizer = None
        self.model = None
        self.name = pth
        self.args = args
        self.tensor = ''
        if pth: self._load(args, pth)
    
    def _load(self, args, pth_path):
        # Load trained model and tokenizer
        label_dict = {'llama1-7b': 18, 'llama1-13b': 22, 'llama1-30b': 32, 'llama1-65b': 42, 'llama3-8b': 18}
        args.num_labels = label_dict[args.model]
        self.args = args

        if args.classifier != 'emb':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.classifier)
            self.model = RobertaForSequenceClassification.from_pretrained(args.classifier, num_labels=args.num_labels)
            self.model.load_state_dict(torch.load(pth_path))
            self.model.cuda()
        else:
            import pdb; pdb.set_trace()
            # 加载question list
            q_path = os.path.join(args.data_path, 'train/train.csv')
            with open(q_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:
                        first_column.append(row[0])
            
            model_path = '/sentence-transformers'
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.tensor = self.model.encode(question_list, convert_to_tensor=True)
            
    def predict(self, LlmModel, context):
        if not self.model:
            return None
        
        int_to_label = {i: label for i, label in enumerate(range(-1, self.args.num_labels))}
        self.model.eval()

        with torch.no_grad():
            if self.args.classifier != 'head':
                inputs = self.tokenizer(context, 
                                            truncation=True,
                                            max_length=self.args.max_len, 
                                            return_tensors='pt').to('cuda')
                
                outputs = self.model(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'])

                classify_prob = outputs.logits.softmax(dim=-1)
                pred = torch.argmax(classify_prob, dim=-1).item()
                return int_to_label[pred]
            
            else:
                last_hidden = get_last_hidden(self.args, context, LlmModel)
                outputs = self.model(last_hidden)
                classify_prob = outputs.softmax(dim=-1)
                pred = torch.argmax(classify_prob, dim=-1).item()
                return int_to_label[pred]

def test_single(args, Classifier, LlmModel, Dataset):
    EvalClass = getattr(eval, f"eval_{args.dataset}")
    EvalClass(args, Dataset, LlmModel, Classifier)


def main(args):
    # create dataloader
    LoaderClass = getattr(test_loader, f'{args.dataset}Loader')
    Dataset = LoaderClass(args)
    
    # load model
    ModelClass = getattr(model, args.model.replace('-', '_'))
    LlmModel = ModelClass(args)
    
    if args.classifier_pths:
        if '.pth' in args.classifier_pths:
            for pth in args.classifier_pths.split(','):
                Classifier = classifier(args, pth)
                args = Classifier.args
                test_single(args, Classifier, LlmModel, Dataset)

        else: # folder
            args.test_folder = True
            pth_files = glob.glob(os.path.join(args.classifier_pths, '*.pth'))
            for file_path in pth_files:
                pth = os.path.abspath(file_path)
                Classifier = classifier(args, pth)
                args = Classifier.args
                test_single(args, Classifier, LlmModel, Dataset)
                # file_name = os.path.basename(file_path)


    elif args.classifier_pth:
        Classifier = classifier(args, args.classifier_pth)
        args = Classifier.args
        test_single(args, Classifier, LlmModel, Dataset)

    else:
        # both 'base' and 'gold' donot need Classifier
        test_single(args, None, LlmModel, Dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['MMLU', 'BBH', 'Gsm8k', 'Comqa', 'StraQA', 'MathQA', 'Reclor', 'PiQA', 'LogiQA', 'Folio', 'AIME', 'ARC_C'], type=str, help='Name of the dataset loader to use')
    parser.add_argument('--model', required=True, choices=['llama1-7b', 'llama1-13b', 'llama1-30b', 'llama1-65b', 'llama3-8b'], type=str, help='Name of the model to use')
    parser.add_argument('--data-path', required=True, type=str, help='Path of valid or test dataset')
    parser.add_argument('--eval-type', required=True, type=str, choices=['log_likelihood', 'base_log', 'base_gen', 'generate'], help='eval modes')
    parser.add_argument('--use-gold', action='store_true', help='use generated layer to eval llama')
    parser.add_argument('--result-path', type=str, default='', help='Path of results')   

    # classifier
    parser.add_argument('--classifier-pth', type=str, default='', help='classifier pth after training')
    parser.add_argument('--classifier-pths', type=str, default='', help='Test all classifier in list')
    parser.add_argument('--test-folder', type=bool, default=False, help='test with classifiers in the folder')
    parser.add_argument('--classifier', default='', type=str, help='classifier model')
    parser.add_argument('--num-labels', default=18, type=int, help='eg. 18 classes for llama-7b')
    parser.add_argument('--max-len', default=512, type=int, help='max length for classifier')

    # special
    parser.add_argument('--task', default='', help='Assign specific tasks in MMLU dataset')
    args = parser.parse_args()

    main(args)
