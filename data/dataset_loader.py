import os
import json
import pandas as pd
import re

def MMLULoader(args):
    # loading raw data
    print("Loading MMLU...")
    TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']
    ans2idx = {'A':1, 'B':2, 'C':3, 'D':4}
    
    if not args.task:
        TASKS = TASKS
    else:
        TASKS = args.task.split(',')

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        label = df_sample.iloc[-1]
        # context = df_sample.iloc[0] + 'The answer is: ' 不要加'The answer is: '
        context = df_sample.iloc[0]
        answer_true = str(df_sample.iloc[ans2idx[label]])
        answer_false = [str(df_sample[i]) for i in [1, 2, 3, 4] if i != ans2idx[label]]
        assert len(answer_false) == 3 and answer_true, "true为空或false个数不为3！"
        return context, answer_true, answer_false
    
    records = {}
    for task in TASKS:
        record = []
        test_df = pd.read_csv(os.path.join('./data/raw_data/MMLU', task + "_test.csv"), header=None)

        for i in range(test_df.shape[0]):
            context, answer_true, answer_false = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
            record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false})
        records[task] = record
        # 当前task数据加载完毕，在records[task]中
    print("Success.\n")
    return records

def Gsm8kLoader(args):
    list_data_dict = []

    def clean(answer):
        answer = re.sub(r'<<.*?>>', '', answer)
        # 替换所有的\n为. 
        answer = answer.replace('\n', ' ')
        # 把####替换为The answer is
        answer = answer.replace('####', 'The answer is')

        return answer + '.'
    
    with open('./data/raw_data/Gsm8k/gsm8k_test.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                question=item['question'],
                answer=clean(item['answer']))
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def FactorLoader(args):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    '''
    Data format:

    ,full_prefix,doc_id,completion,contradiction_0,contradiction_1,contradiction_2,longest_completions,turncated_prefixes
    0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. ",0,Whether or not it gets a second season of The Witcher is another question.,Whether or not it gets a second season of Stranger Things is another question.,Whether or not it gets a fifth season of The Witcher is another question.,Whether or not it gets a second season of Black Mirror is another question.,15.0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. "

    '''
    list_data_dict = []
    df = pd.read_csv(args.data_path)
    prefix_type = 'turncated_prefixes'

    for idx in range(len(df)):
        item = dict(
            context=df[prefix_type][idx],
            answer_true=' '+df['completion'][idx],
            answer_false=[' ' + df[f'contradiction_{i}'][idx] for i in range(3)]
        )
        list_data_dict.append(item)

    # for idx in range(len(df)):
    #     item = dict(
    #         context=df['Question'][idx],
    #         answer_true=' '+df['True_choice'][idx],
    #         answer_false=[' ' + df[f'False_choice{i}'][idx] for i in range(1,4)]
    #     )
    #     list_data_dict.append(item)

    return list_data_dict

def TruthfulQALoader(args):

    list_data = []
    with open(args.data_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data

def BBHLoader(args):
    MULTIPLE_CHOICE_TASKS = [
        "temporal_sequences",
        "disambiguation_qa",
        "date_understanding",
        "tracking_shuffled_objects_three_objects",
        "penguins_in_a_table",
        "geometric_shapes",
        "snarks",
        "ruin_names",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_five_objects",
        "logical_deduction_three_objects",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "movie_recommendation",
        "salient_translation_error_detection",
        "reasoning_about_colored_objects",
    ]
    ans2idx = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17}
    if not args.task:
        TASKS = MULTIPLE_CHOICE_TASKS
    else:
        TASKS = args.Task.split(',')

    def reconstruct(task, ops, target):
        label = target[1]
        if label not in ans2idx.keys():
            print(f"WARNING: Dataset items wrong in task {task}: {target}")
            return None, None
        answer_false = []
        for i in range(len(ops)):
            ops[i] = ops[i][3:]

            if i == ans2idx[label]:
                answer_true = ops[i]
                continue
            answer_false.append(ops[i])
        return answer_true, answer_false
    
    records = {}
    for task in TASKS:
        record = []
        task_data = json.load(open('./data/raw_data/BBH/' + str(task) + '.json'))
        for item in task_data["examples"]:
            item_all = item['input'].split('\n')
            options_idx = item_all.index('Options:')
            if not options_idx:
                import pdb; pdb.set_trace()
            context = '\n'.join(item_all[:options_idx])
            options = item_all[options_idx+1:]

            answer_true, answer_false = reconstruct(task, options, item['target'])
            # 丢弃错误数据item
            if answer_true and answer_false:
                answer_true = answer_true
                answer_false = answer_false
                record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false})
        records[task] = record
        # 当前task数据加载完毕，在records[task]中
    return records

def ComqaLoader(args):
    list_data_dict = []

    with open('./raw_data/Comqa/Comqa_train_rand_split.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            label = item['answerKey']
            context = item['question']['stem']
            choices = item['question']['choices']

            answer_false = []
            for choice in choices:
                if choice['label'] == label:
                    answer_true = choice['text']
                else:
                    answer_false.append(choice['text'])

            new_item = {'context':context, 'answer_true':answer_true, 'answer_false': answer_false}
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def StraQALoader(args):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    list_data_dict = []

    with open('./data/raw_data/StraQA/strategyqa_train.json', 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                term=item.get('term', None),
                description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                facts=item.get('facts', []),
                decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def MathQALoader(args):
    list_data_dict = []

    with open('./data/raw_data/MathQA/test.json', 'r') as f:
        items_test = json.load(f)
    with open('./data/raw_data/MathQA/dev.json', 'r') as f:
        items_dev = json.load(f)

    def reconstruct(options, answer):
        ans2idx = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6}
        
        answers = [re.sub(r'^[a-e]\s*\)\s*', '', part.strip()) for part in options.split(',')]
        answer_false = []
        for i in range(len(answers)):
            if i == ans2idx[answer]:
                answer_true = answers[i]
            else:
                answer_false.append(answers[i])
        return answer_true, answer_false

    items = items_test + items_dev
    for item in items:
        question = item.get('Problem', None)
        options = item.get('options', None)
        answer = item.get('correct', None)

        answer_true, answer_false = reconstruct(options, answer)
        new_item = dict(
            context=question,
            answer_true=answer_true,
            answer_false=answer_false
        )
        list_data_dict.append(new_item)

    return list_data_dict

def ARC_CLoader(args):
    list_data_dict = []

    with open('./data/raw_data/ARC-C/test.json', 'r') as f:
        items_test = json.load(f)
    with open('./data/raw_data/ARC-C/train.json', 'r') as f:
        items_train = json.load(f)

    def reconstruct(options, answer):
        ans2idx = {'A':0, 'B':1, 'C':2, 'D':3, '1':0, '2':1, '3':2, '4':3}
        
        answer_false = []
        for i in range(len(options)):
            if i == ans2idx[answer]:
                answer_true = options[i]
            else:
                answer_false.append(options[i])
        return answer_true, answer_false

    items = items_test + items_train
    for item in items:
        question = item.get('question', None)
        options = item.get('choices', None)['text']
        answer = item.get('answerkey', None)

        answer_true, answer_false = reconstruct(options, answer)
        new_item = dict(
            context=question,
            answer_true=answer_true,
            answer_false=answer_false
        )
        list_data_dict.append(new_item)

    return list_data_dict

def ReclorLoader(args):
    list_data_dict = []

    with open('./data/raw_data/reclor/train.json', 'r') as f:
        items_test = json.load(f)
    with open('./data/raw_data/reclor/val.json', 'r') as f:
        items_dev = json.load(f)

    def reconstruct(options, answer):
        answer_false = []
        for i in range(len(options)):
            if i == answer:
                answer_true = options[i]
            else:
                answer_false.append(options[i])
        return answer_true, answer_false

    items = items_test + items_dev
    for item in items:
        question = item.get('context', None) + ' ' + item.get('question', None)
        options = item.get('answers', None)
        answer = item.get('label', None)

        answer_true, answer_false = reconstruct(options, answer)
        new_item = dict(
            context=question,
            answer_true=answer_true,
            answer_false=answer_false
        )
        list_data_dict.append(new_item)

    return list_data_dict

def PiQALoader(args):
    list_data_dict = []

    with open('./data/raw_data/PiQA/train.jsonl', 'r') as f:
        with open('./data/raw_data/PiQA/train-labels.lst') as g:
            labels_lst = [int(i[0]) for i in g.readlines()]
            i = 0
            for line in f:
                item = json.loads(line)
                if labels_lst[i]:
                    new_item = dict(
                        context=item['goal'],
                        answer_true=item['sol2'],
                        answer_false=[item['sol1']])
                else:
                    new_item = dict(
                        context=item['goal'],
                        answer_true=item['sol1'],
                        answer_false=[item['sol2']])

                item = new_item
                list_data_dict.append(item)
                i += 1

    with open('./data/raw_data/PiQA/valid.jsonl', 'r') as f:
        with open('./data/raw_data/PiQA/valid-labels.lst') as g:
            labels_lst = [int(i[0]) for i in g.readlines()]
            i = 0
            for line in f:
                item = json.loads(line)

                if labels_lst[i]:
                    new_item = dict(
                        context=item['goal'],
                        answer_true=item['sol2'],
                        answer_false=[item['sol1']])
                else:
                    new_item = dict(
                        context=item['goal'],
                        answer_true=item['sol1'],
                        answer_false=[item['sol2']])

                item = new_item
                list_data_dict.append(item)
                i += 1

    return list_data_dict

def LogiQALoader(args):
    list_data_dict = []

    with open('./data/raw_data/LogiQA/Train.txt', 'r') as f:
        train = f.readlines()
    with open('./data/raw_data/LogiQA/Eval.txt', 'r') as f:
        valid = f.readlines()
    data = train + valid

    i = 1
    while i < len(data):
        # 获取标准答案
        answer_true = data[i].strip()
        i += 1

        # 获取context
        context = data[i].strip()
        i += 1

        # 获取问题
        question = data[i].strip()
        i += 1

        # 获取四个选项
        options = []
        for _ in range(4):
            options.append(data[i].strip())
            i += 1

        # 获取错误选项
        # import pdb; pdb.set_trace()
        answer_false = []
        for option in options:
            # import pdb; pdb.set_trace()
            if option.startswith(answer_true.upper() + '.'):
                answer_true = option[2:]
            else:
                answer_false.append(option[2:])

        # 合并问题和上下文
        full_context = f"{context} {question}"

        # 构建字典
        data_entry = {
            "context": full_context,
            "answer_true": answer_true,
            "answer_false": answer_false
        }

        # 添加到数据集中
        list_data_dict.append(data_entry)

        # 跳过换行
        i += 1
    # import pdb; pdb.set_trace()
    return list_data_dict

def FolioLoader(args):
    list_data_dict = []

    with open('./data/raw_data/Folio/folio-train.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                context=' '.join(item['premises']),
                question=item['conclusion'],
                answer=str(item['label']), 
                fol=', '.join(item['premises-FOL']) + '. ')

            item = new_item
            list_data_dict.append(item)

    with open('./data/raw_data/Folio/folio-validation.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                context=' '.join(item['premises']),
                question=item['conclusion'],
                answer=str(item['label']), 
                fol=', '.join(item['premises-FOL']) + '. ')

            item = new_item
            list_data_dict.append(item)
    
    # import pdb; pdb.set_trace()
    return list_data_dict