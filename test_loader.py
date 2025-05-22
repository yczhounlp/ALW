# 文件名：dataset_loader.py
import os
import re
import ast
import json
import pandas as pd

def MMLULoader(args):
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
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = [str(df_sample[i]) for i in [2, 3, 4]]
        assert len(answer_false) == 3 and answer_true, "true为空或false个数不为3！"
        return context, answer_true, answer_false, best_layer
    
    records = {}
    for task in TASKS:
        record = []
        test_df = pd.read_csv(os.path.join(args.data_path, f"test/{task}.csv"), header=None)
        
        for i in range(1, test_df.shape[0]): # skip first row
            context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
            record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})
            # import pdb; pdb.set_trace()
        records[task] = record
        # 当前task数据加载完毕，在records[task]中
    print("Success.\n")
    return records

def Gsm8kLoader(args):
    print("Loading Gsm8k...")

    list_data_dict = []

    def clean(answer):
        ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
        match = ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(".", "")
            return match_str
        else:
            return "[invalid]"
    
    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        question, answer = df.iloc[i]

        new_item = dict(
            question=question,
            answer=clean(answer))
        item = new_item
        list_data_dict.append(item)

    return list_data_dict


def BBHLoader(args):
    print("Loading BBH...")
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
    
    if not args.task:
        TASKS = MULTIPLE_CHOICE_TASKS
    else:
        TASKS = args.task.split(',')

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    for task in TASKS:
        record = []
        test_df = pd.read_csv(os.path.join(args.data_path, f"test/{task}.csv"), header=None)
        
        for i in range(1, test_df.shape[0]): # skip first row
            context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
            record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})
            # import pdb; pdb.set_trace()
        records[task] = record
        # 当前task数据加载完毕，在records[task]中
    print("Success.\n")
    return records

def ComqaLoader(args):
    print("Loading Comqa...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})
        # import pdb; pdb.set_trace()
    records['Comqa'] = record
    # 和MMLU、BBH统一格式，虽然Comqa只有一个数据集但也写成records[task]的形式
    print("Success.\n")
    return records

def StraQALoader(args):
    print("Loading StraQA...")

    list_data_dict = []

    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        _, _, _, question, answer, _, _ = df.iloc[i]

        new_item = dict(
            question=question,
            answer=str(answer))
        item = new_item
        list_data_dict.append(item)

    return list_data_dict

def MathQALoader(args):
    print("Loading MathQA...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})

    records['MathQA'] = record
    # 和ComQA基本相同
    print("Success.\n")
    return records


def ReclorLoader(args):
    print("Loading Reclor...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})

    records['Comqa'] = record
    # 和ComQA基本相同
    print("Success.\n")
    return records

def PiQALoader(args):
    print("Loading PiQA...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})

    records['PiQA'] = record
    # 和ComQA基本相同
    print("Success.\n")
    return records

def ARC_CLoader(args):
    print("Loading ARC_C...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})

    records['ARC_C'] = record
    # 和ComQA基本相同
    print("Success.\n")
    return records

def LogiQALoader(args):
    print("Loading LogiQA...")

    def reconstruct(df_sample):
        '''
        input:
        df数据中采样的一行，# df.iloc[i][j] 获取第i行第j列的数据
        return:
        context: str, answer_true: str, answer_false: list(str*3)
        '''
        best_layer = int(df_sample.iloc[-1])

        context = df_sample[0]
        answer_true = str(df_sample[1])
        answer_false = ast.literal_eval(df_sample[2])
        return context, answer_true, answer_false, best_layer
    
    records = {}
    record = []
    test_df = pd.read_csv(os.path.join(args.data_path, f"test/test.csv"), header=None)
    
    for i in range(1, test_df.shape[0]): # skip first row
        context, answer_true, answer_false, best_layer = reconstruct(test_df.iloc[i]) # 直接test_df[i]是返回列
        record.append({'context':context, 'answer_true':answer_true, 'answer_false': answer_false, 'best_layer': best_layer})

    records['LogiQA'] = record
    # 和ComQA基本相同
    print("Success.\n")
    return records

def FolioLoader(args):
    print("Loading Folio...")
    list_data_dict = []

    df = pd.read_csv(os.path.join(args.data_path, 'test/test.csv'), header=None)
    for i in range(1, df.shape[0]): # skip first row
        context,question,answer,fol = df.iloc[i]

        new_item = dict(
            context=context,
            question=question,
            answer=str(answer),
            fol=fol)
        item = new_item
        list_data_dict.append(item)

    return list_data_dict