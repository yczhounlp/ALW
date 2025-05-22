import json
from tqdm import tqdm
import numpy as np
from make_type import log_likelihood, generate
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
import h5py
import torch

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

    return logits[mature_layer][0][-1].cpu()

# 对于一个数据集有多个子数据集的情况，由于test需要每个数据集单独切割不能合并
# 所以会先暂存在all文件夹中，在train阶段分情况切割
def make_MMLU_train(args, Dataset, Model):
    print(f'Multichoice Task: Making training data in {args.dataset}.')
    
    for task in Dataset.keys():
        write_dict = {'context': [], 'answer_true': [], 'answer_false0': [], 'answer_false1': [], 'answer_false2': [], 'best_layer': []}
        print('Making %s ...' % task)

        for sample in tqdm(Dataset[task]):
            answer_true_log_probs = np.array(log_likelihood(args, Model, sample['context'], sample['answer_true']))

            false_log_prob0 = log_likelihood(args, Model, sample['context'], sample['answer_false'][0])
            false_log_prob1 = log_likelihood(args, Model, sample['context'], sample['answer_false'][1])
            false_log_prob2 = log_likelihood(args, Model, sample['context'], sample['answer_false'][2])
            false_log_probs = list(zip(false_log_prob0, false_log_prob1, false_log_prob2))
            # 实时评测，而不是生成完再评测
            false_max_probs = np.array([max(prob) for prob in false_log_probs])
            answer = ~(false_max_probs > answer_true_log_probs)

            filtered_array = answer_true_log_probs[answer]
            if filtered_array.size:
                max_index = np.argmax(filtered_array)
                original_indices = np.arange(len(answer))[answer]
                best_layer = original_indices[max_index]

            else:
                best_layer = np.argmax(answer_true_log_probs)

            # 如果final layer可以做对且和其他层并列最大
            # 如果所有层都不能做对
            # 都优先选择final layer
            if answer_true_log_probs[best_layer] == answer_true_log_probs[-1]:
                best_layer = -1

            # import pdb; pdb.set_trace()
            write_dict['context'].append(sample['context'])
            write_dict['answer_true'].append(sample['answer_true'])
            write_dict['answer_false0'].append(sample['answer_false'][0])
            write_dict['answer_false1'].append(sample['answer_false'][1])
            write_dict['answer_false2'].append(sample['answer_false'][2])
            write_dict['best_layer'].append(best_layer)

        df = pd.DataFrame(write_dict)
        output_folder = os.path.join('./data/training_data', args.model + '/MMLU/all')

        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(os.path.join(output_folder, task+'.csv'), index=False)

def make_Gsm8k_train(args, Dataset, Model):
    print(f'Generate Task: Making training data in {args.dataset}.')
    train_path = os.path.join('./data/training_data', args.model, args.dataset, 'train')
    valid_path = os.path.join('./data/training_data', args.model, args.dataset, 'valid')
    test_path = os.path.join('./data/training_data', args.model, args.dataset, 'test')

    for directory in [train_path, valid_path, test_path]:
        os.makedirs(directory, exist_ok=True)

    Dataset_df = pd.DataFrame(Dataset)

    train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
    test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
    
    def Gsm8k_create_demo_text(n_shot=8):
        question, chain, answer = [], [], []
        question.append("There are 15 trees in the grove. "
                        "Grove workers will plant trees in the grove today. "
                        "After they are done, there will be 21 trees. "
                        "How many trees did the grove workers plant today?")
        chain.append("There are 15 trees originally. "
                    "Then there were 21 trees after some more were planted. "
                    "So there must have been 21 - 15 = 6.")
        answer.append("6")

        question.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?")
        chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        answer.append("5")

        question.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?")
        chain.append("Originally, Leah had 32 chocolates. "
                    "Her sister had 42. So in total they had 32 + 42 = 74. "
                    "After eating 35, they had 74 - 35 = 39.")
        answer.append("39")

        question.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny?")
        chain.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some "
            "to Denny. So he gave Denny 20 - 12 = 8.")
        answer.append("8")

        question.append(
            "Shawn has five toys. For Christmas, he got two toys each from his "
            "mom and dad. How many toys does he have now?")
        chain.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and "
            "dad, then that is 4 more toys. 5 + 4 = 9.")
        answer.append("9")

        question.append(
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from monday to thursday. "
            "How many computers are now in the server room?")
        chain.append(
            "There were originally 9 computers. For each of 4 days, 5 more "
            "computers were added. So 5 * 4 = 20 computers were added. "
            "9 + 20 is 29.")
        answer.append("29")

        question.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
            "wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?")
        chain.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, "
            "he had 58 - 23 = 35. After losing 2 more, "
            "he had 35 - 2 = 33 golf balls.")
        answer.append("33")

        question.append("Olivia has $23. She bought five bagels for $3 each. "
                        "How much money does she have left?")
        chain.append("Olivia had 23 dollars. "
                    "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                    "So she has 23 - 15 dollars left. 23 - 15 is 8.")
        answer.append("8")

        index_list = list(range(len(question)))

        ANSWER_TRIGGER = "The answer is"
        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list[:n_shot]:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                        ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
            
        return demo_text

    write_dict = {'context': [], 'best_layer': []}
    for sample in tqdm(train_valid_data.to_dict(orient='records')):
        shot = Gsm8k_create_demo_text(n_shot=8)

        context, best_layer = generate(args, Model, shot, "Q: " + sample['question'] + "\n" + "A:", sample['answer'])
        write_dict['context'] += context
        write_dict['best_layer'] += best_layer
    
    df = pd.DataFrame(write_dict)
    train_data, valid_data = train_test_split(df, test_size=0.2, random_state=42)

    train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)


def make_BBH_train(args, Dataset, Model):
    print(f'Multichoice Task: Making training data in {args.dataset}.')
    
    for task in Dataset.keys():
        print('Making %s ...' % task)
        write_dict = {'context': [], 'answer_true': [], 'answer_false': [], 'best_layer': []}

        for sample in tqdm(Dataset[task]):
            answer_true_log_probs = np.array(log_likelihood(args, Model, sample['context'], sample['answer_true']))
            false_log_probs = [np.array(log_likelihood(args, Model, sample['context'], answer_false)) for answer_false in sample['answer_false']]
            
            false_max_probs = np.maximum.reduce([option for option in false_log_probs])
            answer = ~(false_max_probs > answer_true_log_probs)

            filtered_array = answer_true_log_probs[answer]
            if filtered_array.size:
                max_index = np.argmax(filtered_array)
                original_indices = np.arange(len(answer))[answer]
                best_layer = original_indices[max_index]

            else:
                best_layer = np.argmax(answer_true_log_probs)

            # 如果final layer可以做对且和其他层并列最大
            # 如果所有层都不能做对
            # 都优先选择final layer
            if answer_true_log_probs[best_layer] == answer_true_log_probs[-1]:
                best_layer = -1

            # import pdb; pdb.set_trace()
            write_dict['context'].append(sample['context'])
            write_dict['answer_true'].append(sample['answer_true'])
            write_dict['answer_false'].append(sample['answer_false'])
            write_dict['best_layer'].append(int(best_layer))
        
        output_folder = os.path.join('./data/training_data', args.model + '/BBH/all')
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame(write_dict)
        df.to_csv(os.path.join(output_folder, task+'.csv'), index=False)

def make_Comqa_train(args, Dataset, Model):
    print(f'Multichoice Task: Making training data in {args.dataset}.')
    write_dict = {'context': [], 'answer_true': [], 'answer_false': [], 'best_layer': []}
    train_path = os.path.join('./data/training_data', args.model, args.dataset, 'train')
    valid_path = os.path.join('./data/training_data', args.model, args.dataset, 'valid')
    test_path = os.path.join('./data/training_data', args.model, args.dataset, 'test')
    if args.head:
        tensor_file = os.path.join(train_path, 'data.h5')
        
    for directory in [train_path, valid_path, test_path]:
        os.makedirs(directory, exist_ok=True)

    tensor_data, label_data, text_data = [], [], []
    # i = 0
    for sample in tqdm(Dataset):
        # import pdb; pdb.set_trace()
        answer_true_log_probs = np.array(log_likelihood(args, Model, sample['context'], sample['answer_true']))
        false_log_probs = [np.array(log_likelihood(args, Model, sample['context'], answer_false)) for answer_false in sample['answer_false']]
        
        false_max_probs = np.maximum.reduce([option for option in false_log_probs])
        answer = ~(false_max_probs > answer_true_log_probs)

        filtered_array = answer_true_log_probs[answer]
        if filtered_array.size:
            max_index = np.argmax(filtered_array)
            original_indices = np.arange(len(answer))[answer]
            best_layer = original_indices[max_index]

        else:
            best_layer = np.argmax(answer_true_log_probs)

        # 如果final layer可以做对且和其他层并列最大
        # 如果所有层都不能做对
        # 都优先选择final layer
        if answer_true_log_probs[best_layer] == answer_true_log_probs[-1]:
            best_layer = -1

        # import pdb; pdb.set_trace()
        write_dict['context'].append(sample['context'])
        write_dict['answer_true'].append(sample['answer_true'])
        write_dict['answer_false'].append(sample['answer_false'])
        write_dict['best_layer'].append(int(best_layer))
        
        if args.head:
            tensor_data.append(get_last_hidden(args, sample['context'], Model).numpy())
            text_data.append(sample['context'].encode('utf8'))
            label_data.append(int(best_layer))
        # i += 1
        # if i == 10:
        #     break

    # 写入csv
    df = pd.DataFrame(write_dict)
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)
    test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)

    # 写入h5
    if args.head:
        split_idx = np.array(train_data.index.tolist())
        tensor_data = np.array(tensor_data)[split_idx]
        text_data = [i.decode('utf-8') for i in np.array(text_data)[split_idx]]
        label_data = np.array(label_data)[split_idx]
        
        with h5py.File(tensor_file, 'w') as f:
            f.create_dataset('tensor', data=tensor_data, compression='gzip', compression_opts=5)
            f.create_dataset('texts', data=text_data, compression='gzip', compression_opts=5)
            f.create_dataset('labels', data=label_data, compression='gzip', compression_opts=5)


def make_StraQA_train(args, Dataset, Model):
    print(f'Generate Task: Making training data in {args.dataset}.')
    train_path = os.path.join('./data/training_data', args.model, args.dataset, 'train')
    valid_path = os.path.join('./data/training_data', args.model, args.dataset, 'valid')
    test_path = os.path.join('./data/training_data', args.model, args.dataset, 'test')

    for directory in [train_path, valid_path, test_path]:
        os.makedirs(directory, exist_ok=True)

    Dataset_df = pd.DataFrame(Dataset)

    train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
    test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
    
    def StraQA_create_demo_text(n_shot=6):
        QUESTION_TRIGGER = "\nLet's think step by step. "
        ANSWER_TRIGGER = "So the answer is"
        question, chain, answer = [], [], []

        question.append("Do hamsters provide food for any animals?")
        chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
        answer.append("yes")

        question.append("Could Brooke Shields succeed at University of Pennsylvania?")
        chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
        answer.append("yes")

        question.append("Hydrogen's atomic number squared exceeds number of Spice Girls?")
        chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
        answer.append("no")

        question.append("Is it common to see frost during some college commencements?")
        chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
        answer.append("yes")
        
        question.append("Could a llama birth twice during War in Vietnam (1945-46)?")
        chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
        answer.append("no")

        question.append("Would a pear sink in water?")
        chain.append("The density of a pear is about 0.59 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
        answer.append("no")

        # randomize order of the examples ...
        demo_text = ''
        index_list = list(range(len(question)))

        for i in index_list[:n_shot]:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
            
        return demo_text

    write_dict = {'context': [], 'best_layer': []}
    for sample in tqdm(train_valid_data.to_dict(orient='records')):
        shot = StraQA_create_demo_text(n_shot=6)

        facts = '. '.join([fact.strip()for fact in sample['facts']])
        answer = 'yes' if sample['answer'] else 'no'
        context, best_layer = generate(args, Model, shot, "Q: " + sample['question'] + "\n" + "A:", facts + " So the answer is " + answer)
        write_dict['context'] += context
        write_dict['best_layer'] += best_layer
    
    df = pd.DataFrame(write_dict)
    train_data, valid_data = train_test_split(df, test_size=0.2, random_state=42)

    train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)

def make_Folio_train(args, Dataset, Model):
    print(f'Generate Task: Making training data in {args.dataset}.')
    train_path = os.path.join('./data/training_data', args.model, args.dataset, 'train')
    valid_path = os.path.join('./data/training_data', args.model, args.dataset, 'valid')
    test_path = os.path.join('./data/training_data', args.model, args.dataset, 'test')

    for directory in [train_path, valid_path, test_path]:
        os.makedirs(directory, exist_ok=True)

    Dataset_df = pd.DataFrame(Dataset)

    train_valid_data, test_data = train_test_split(Dataset_df, test_size=0.1, random_state=42)
    test_data.to_csv(os.path.join(test_path, 'test.csv'), index=False)
    
    def Folio_create_demo_text(n_shot=3):
        context, question, chain, answer = [], [], [], []
        context.append("All kids love animals. All toddlers are kids. If someone loves animals, then they are not bad. "
                        "All pirates are bad. If Nancy is not a pirate, then Nancy loves animals. "
                        "If Nancy is not a toddler, then Nancy is bad.")
        question.append('Nancy is a pirate.')
        chain.append("∀x (Kid(x) → LoveAnimals(x)), ∀x (Toddler(x) → Kid(x)), "
                     "∀x (LoveAnimals(x) → ¬Bad(x)), ∀x (Pirate(x) → Bad(x)), "
                     "¬Pirate(nancy) → LoveAnimals(nancy), ¬Toddler(nancy) → Bad(nancy). ")
        answer.append("Unknown")

        context.append("Oxford Circus is a road junction connecting Oxford Street and Regent Street. "
                       "Oxford Street is in London. John Nash designed the construction on Regent Street. "
                       "John Nash designed Oxford Circus. John Nash is an architect in British. "
                       "Oxford Circus is the entrance to Oxford Circus tube station which is a part of the Central line in 1900.")
        question.append('Oxford Circus is in London.')
        chain.append("OxfordStreet(oxfordCircus) ∧ RegentStreet(oxfordCircus), ∀x (OxfordStreet(x) → London(x)), "
                     "∃x (RegentStreet(x) ∧ Design(johnNash, x)), Design(johnNash, oxfordCircus), "
                     "British(johnNash) ∧ Architect(johnNash), CentralLine(oxfordCircus). ")
        answer.append("True")

        context.append("Buisnesses are either sanctioned or unsanctioned. Sanctioned buisnesses are limited. "
                       "Unsanctioned buisnesses are free. The Crude Oil Data Exchange is a buisness that isn't free. ")
        question.append('Crude Oil Data Exchange is unsanctioned.')
        chain.append("∀x (Buisness(x) → Sanctioned(x) ⊕ ¬Sanctioned(x)), ∀x (Buisness(x) ∧ Sanctioned(x) → Limited(x)), "
                     "∀x (Buisness(x) ∧ ¬Sanctioned(x) → Free(x)), Buisness(crudeoildataexchange) ∧ ¬Free(crudeoildataexchange)")
        answer.append("False")

        index_list = list(range(len(context)))

        ANSWER_TRIGGER = "The statement is"
        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list[:n_shot]:
            demo_text += "Context: " + context[i] + "\nQuestion: The statement " + '\'' + question[i] + '\'' + " is True, False or Unknown?\nA: " + \
                        chain[i] + ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
           
        return demo_text

    write_dict = {'context': [], 'best_layer': []}
    for sample in tqdm(train_valid_data.to_dict(orient='records')):
        shot = Folio_create_demo_text(n_shot=3)
        # import pdb; pdb.set_trace()
        context, best_layer = generate(args, Model, shot, "Context: " + sample['context'] + "\nQuestion: The statement " + '\'' + sample['question'] + '\'' + " is True, False or Unknown?\nA: ", sample['fol'] + "The statement is "  + sample['answer'] + ".")
        write_dict['context'] += context
        write_dict['best_layer'] += best_layer
    
    df = pd.DataFrame(write_dict)
    train_data, valid_data = train_test_split(df, test_size=0.2, random_state=42)

    train_data.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    valid_data.to_csv(os.path.join(valid_path, 'valid.csv'), index=False)

def make_MathQA_train(args, Dataset, Model):
    make_Comqa_train(args, Dataset, Model)

def make_Reclor_train(args, Dataset, Model):
    make_Comqa_train(args, Dataset, Model)

def make_PiQA_train(args, Dataset, Model):
    make_Comqa_train(args, Dataset, Model)
    
def make_ARC_C_train(args, Dataset, Model):
    make_Comqa_train(args, Dataset, Model)

def make_LogiQA_train(args, Dataset, Model):
    make_Comqa_train(args, Dataset, Model)