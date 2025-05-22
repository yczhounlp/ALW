import eval_type
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import json
import time
import os
import re

# def forward_classifier(args, Classifier, content):
#     if not Classifier:
#         return None
    
#     int_to_label = {i: label for i, label in enumerate(range(-1, args.num_labels))}
#     Classifier.model.eval()

#     with torch.no_grad():
#         inputs = Classifier.tokenizer(content, 
#                                       truncation=True,
#                                       max_length=args.max_len, 
#                                       return_tensors='pt').to('cuda')
        
#         outputs = Classifier.model(input_ids=inputs['input_ids'], 
#                                    attention_mask=inputs['attention_mask'])

#         classify_prob = outputs.logits.softmax(dim=-1)
#         pred = torch.argmax(classify_prob, dim=-1).item()
#         return int_to_label[pred]

def eval_MMLU(args, Dataset, LlmModel, Classifier):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        # 自动分配result path
        # import pdb; pdb.set_trace()
        result_path = os.path.join(args.classifier_pths, os.path.basename(Classifier.name).replace('.pth', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)

    if args.eval_type == 'base_log':
        print(f'Multichoice Task: Eval {args.dataset} with {args.model} without classifier')  
    elif args.use_gold:
        print(f'Multichoice Task: Eval {args.dataset} with {args.model} and gold layer')
    else:
        print(f'Multichoice Task: Eval {args.dataset} with {args.model} and classifier in {Classifier.name}')

    cls_result_dic , llm_result_dic = {}, {}
    total_cls_cor = 0 # classifier correct num
    total_llm_cor = 0 # llm correct num
    total_num = 0

    for task in Dataset.keys():
        cls_cor, llm_cor, num = 0, 0, 0
        write_dict = {'content': [], 'answer_true': [], 'answer_false': [], 'pred_layer': [], 'best_layer': [], 'llm_answer': []}
        print('Evaluating %s ...' % task)

        for sample in tqdm(Dataset[task]):
            # TODO: predict放到eval mode中？还是单独forward？
            # import pdb; pdb.set_trace()
            early_exit_layer = Classifier.predict(LlmModel, sample['context'].strip()) if args.eval_type != 'base_log' else 0
            # import pdb; pdb.set_trace()
            # early_exit_layer = 0 # dola_static
            cls_cor = cls_cor + (early_exit_layer == sample['best_layer']) if args.eval_type != 'base_log' else 0
            # sample['best_layer']).sum().item()
            num += 1

            # eval llm
            true_log_prob = eval_mode(args, LlmModel, sample['context'], sample['answer_true'], early_exit_layer)
            # false_log_probs = [eval_mode(args, LlmModel, sample['context'], sample['answer_false'][i], early_exit_layer) for i in [0, 1, 2]]
            false_log_probs = [eval_mode(args, LlmModel, sample['context'], answer_false, early_exit_layer) for answer_false in sample['answer_false']]
            is_cor = True
            for i in false_log_probs:
                if i > true_log_prob:
                    is_cor = False
            llm_cor += is_cor

            write_dict['content'].append(sample['context'])
            write_dict['answer_true'].append(sample['answer_true'])
            write_dict['answer_false'].append(sample['answer_false'])
            write_dict['pred_layer'].append(early_exit_layer)
            write_dict['best_layer'].append(sample['best_layer'])
            write_dict['llm_answer'].append(is_cor)

        cls_result_dic[task] = round(cls_cor/num, 4)
        llm_result_dic[task] = round(llm_cor/num, 4)

        total_llm_cor += llm_cor
        total_cls_cor += cls_cor
        total_num += num

        df = pd.DataFrame(write_dict)
        df.to_csv(os.path.join(result_path, task+'.csv'), index=False)

    cls_result_dic['all'] = round(total_cls_cor/total_num, 4)
    llm_result_dic['all'] = round(total_llm_cor/total_num, 4)

    with open(os.path.join(result_path, "results.txt"), "w") as file:
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4)
        file.write(args_str)
        file.write('\n')

        for dataset, accuracy in llm_result_dic.items():
            file.write(f"{dataset:35}: {accuracy:6}(llm), {cls_result_dic[dataset]:6}(adapter)\n")

def eval_BBH(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_Comqa(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_Gsm8k(args, Dataset, LlmModel, Classifier):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        # 自动分配result path
        result_path = os.path.join(args.classifier_pths, os.path.basename(Classifier.name).replace('.pth', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without classifier')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and classifier in {Classifier.name}')
    
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

    def Gsm8k_clean_answer(model_pred):
        ANSWER_TRIGGER = "The answer is"

        model_pred = model_pred.lower()
        preds = model_pred.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return "[invalid]"
        
        if answer_flag: # choose the first element in list
            pred = pred[0]
        else: # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred
    
    def Gsm8k_is_correct(model_answer, answer):
        assert answer != "[invalid]"
        return model_answer == answer
    
    answers = []
    prediction_lm = {i:0 for i in range(17)}
    prediction_lm[-1] = 0

    for sample in tqdm(Dataset):
        shot = Gsm8k_create_demo_text(n_shot=8)

        # 在生成中每一次预测都要经过classifier
        # TODO: 使用Classifier.predict()函数, eval_mode中少传了最后一个参数
        model_completion, best_layer_dic = eval_mode(args, LlmModel, shot, "Q: " + sample['question'] + "\n" + "A:", Classifier)
        model_completion = model_completion.strip()
        prediction_lm = {k: best_layer_dic.get(k, 0) + prediction_lm.get(k, 0) for k in set(best_layer_dic) | set(prediction_lm)}

        model_answer = Gsm8k_clean_answer(model_completion)
        is_cor = Gsm8k_is_correct(model_answer, sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    print(prediction_lm)
    f.close()
    return

def eval_AIME(args, Dataset, LlmModel, Classifier):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        # 自动分配result path
        result_path = os.path.join(args.classifier_pths, os.path.basename(Classifier.name).replace('.pth', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without classifier')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and classifier in {Classifier.name}')
    
    def AIME_create_demo_text(n_shot=3):
        question, chain, answer = [], [], []
        question.append("Let $N$ be the greatest four-digit positive integer with the property that whenever one of its digits is changed to $1$, the resulting number is divisible by $7$. Let $Q$ and $R$ be the quotient and remainder, respectively, when $N$ is divided by $1000$. Find $Q+R$.")
        chain.append("We note that by changing a digit to $1$ for the number $\\overline{abcd}$, we are subtracting the number by either $1000(a-1)$, $100(b-1)$, $10(c-1)$, or $d-1$. Thus, $1000a + 100b + 10c + d \\equiv 1000(a-1) \\equiv 100(b-1) \\equiv 10(c-1) \\equiv d-1 \\pmod{7}$. We can casework on $a$ backwards, finding the maximum value. (Note that computing $1000 \\equiv 6 \\pmod{7}, 100 \\equiv 2 \\pmod{7}, 10 \\equiv 3 \\pmod{7}$ greatly simplifies computation). Applying casework on $a$, we can eventually obtain a working value of $\\overline{abcd} = 5694 \\implies \\boxed{699}$.")
        answer.append("699")

        question.append("Let ABCDEF be a convex equilateral hexagon in which all pairs of opposite sides are parallel. The triangle whose sides are extensions of segments AB, CD, and EF has side lengths 200, 240, and 300. Find the side length of the hexagon.")
        chain.append("Draw an accurate diagram using the allowed compass and ruler: Draw a to-scale diagram of the $(200,240,300)$ triangle (e.g. 10cm-12cm-15cm). Because of the nature of these lengths and the integer answer needed, it can be assumed that the side length of the hexagon will be divisible by 10. Therefore, a trial-and-error method can be set up, wherein line segments of length $n\\cdot 10$, scaled to the same scale as the triangle, can be drawn to represent the sides of the hexagon. For instance, side $BC$ would be drawn parallel to the side of the triangle of length 300, and would have endpoints on the other sides of the triangle. Using this method, it would be evident that line segments of length 80 units, scaled proportionally (4cm using the scale above), would create a perfect equilateral hexagon when drawn parallel to the sides of the triangle. $x=\\boxed{080}$.")
        answer.append("80")

        question.append("Let $\\omega \\neq 1$ be a 13th root of unity. Find the remainder when \n\\[ \\prod_{k=0}^{12}(2 - 2\\omega^k + \\omega^{2k}) \\] is divided by 1000.")
        chain.append("To find $\\prod_{k=0}^{12} (2 - 2\\omega^k + \\omega^{2k})$, where $\\omega \\neq 1$ and $\\omega^{13} = 1$, rewrite this as \n$(r - \\omega)(s - \\omega)(r - \\omega^2)(s - \\omega^2)...(r - \\omega^{12})(s - \\omega^{12})$ where $r$ and $s$ are the roots of the quadratic $x^2 - 2x + 2 = 0$. \nGrouping the $r$'s and $s$'s results in \n$\\frac{r^{13} - 1}{r - 1} \\cdot \\frac{s^{13} - 1}{s - 1}$. \nThe denominator $(r - 1)(s - 1) = 1$ by Vieta's formulas. \nThe numerator $(rs)^{13} - (r^{13} + s^{13}) + 1 = 2^{13} - (-128) + 1 = 8321$ by Newton's sums. \nThus, the final answer is $\\boxed{321}$.")
        answer.append("321")

        index_list = list(range(len(question)))

        ANSWER_TRIGGER = "The answer is"
        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list[:n_shot]:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                        ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
            
        return demo_text

    def AIME_clean_answer(model_pred):
        ANSWER_TRIGGER = "The answer is"

        model_pred = model_pred.lower()
        preds = model_pred.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return "[invalid]"
        
        if answer_flag: # choose the first element in list
            pred = pred[0]
        else: # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred
    
    def AIME_is_correct(model_answer, answer):
        assert answer != "[invalid]"
        return model_answer == answer
    
    answers = []
    prediction_lm = {i:0 for i in range(17)}
    prediction_lm[-1] = 0

    for sample in tqdm(Dataset):
        shot = AIME_create_demo_text(n_shot=3)

        model_completion, best_layer_dic = eval_mode(args, LlmModel, shot, "Q: " + sample['question'] + "\n" + "A:", Classifier)
        model_completion = model_completion.strip()
        prediction_lm = {k: best_layer_dic.get(k, 0) + prediction_lm.get(k, 0) for k in set(best_layer_dic) | set(prediction_lm)}

        model_answer = AIME_clean_answer(model_completion)
        is_cor = AIME_is_correct(model_answer, sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    print(prediction_lm)
    f.close()
    return

def eval_StraQA(args, Dataset, LlmModel, Classifier):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        # 自动分配result path
        result_path = os.path.join(args.classifier_pths, os.path.basename(Classifier.name).replace('.pth', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    answers = []
    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without classifier')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and classifier in {Classifier.name}')

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

    def StraQA_clean_answer(model_pred):
        SHORT_ANSWER_TRIGGER = "answer is"
        model_pred = model_pred.lower()

        if "Thus, yes." in model_pred:
            preds = "yes"
        elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
            preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
        else:
            print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
            preds = "no" if "no" in model_pred else "yes"

        if preds not in ["yes", "no"]:
            print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
            preds = "no"

        return (preds == "yes")
    
    def StraQA_is_correct(model_answer, answer):
        gt_answer = answer
        assert gt_answer != "[invalid]"
        return model_answer == gt_answer
    
    for sample in tqdm(Dataset):
        shot = StraQA_create_demo_text(n_shot=6)

        model_completion, _ = eval_mode(args, LlmModel, shot, "Q: " + sample['question'] + "\n" + "A:", Classifier)
        model_completion = model_completion.strip()

        model_answer = StraQA_clean_answer(model_completion)
        is_cor = StraQA_is_correct(str(model_answer), sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    f.close()
    return

def eval_MathQA(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_Reclor(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_PiQA(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_LogiQA(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_ARC_C(args, Dataset, LlmModel, Classifier):
    eval_MMLU(args, Dataset, LlmModel, Classifier)

def eval_Folio(args, Dataset, LlmModel, Classifier):
    eval_mode = getattr(eval_type, args.eval_type)
    if args.test_folder:
        # 自动分配result path
        result_path = os.path.join(args.classifier_pths, os.path.basename(Classifier.name).replace('.pth', ''))
    else:
        result_path = args.result_path

    os.makedirs(result_path, exist_ok=True)
    f = open(os.path.join(result_path, 'result.txt'), 'w')

    answers = []
    if args.eval_type == 'base_gen':
        print(f'Generate Task: Eval {args.dataset} with {args.model} without classifier')  
    else:
        print(f'Generate Task: Eval {args.dataset} with {args.model} and classifier in {Classifier.name}')

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

    def Folio_clean_answer(model_pred):
        ANSWER_TRIGGER = "The statement is"

        model_pred = model_pred.lower()
        preds = model_pred.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        # import pdb; pdb.set_trace()
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.strip().replace(".", "")

        if pred == 'true':
            return str(True)
        elif pred == 'false':
            return str(False)
        elif pred == 'unknown':
            return str('Unknown')            
        else:
            return "[invalid]"

        return pred
    
    def Folio_is_correct(model_answer, answer):
        gt_answer = answer
        assert gt_answer != "[invalid]"
        return model_answer == gt_answer
    
    for sample in tqdm(Dataset):
        shot = Folio_create_demo_text(n_shot=3)

        model_completion, _ = eval_mode(args, LlmModel, shot, "Context: " + sample['context'] + "\nQuestion: The statement " + '\'' + sample['question'] + '\'' + " is True, False or Unknown?\nA: ", Classifier)
        model_completion = model_completion.strip()

        model_answer = Folio_clean_answer(model_completion)
        is_cor = Folio_is_correct(str(model_answer), sample['answer'])
        answers.append(is_cor)

        print(f'Question: {sample["question"]}\n\n'
            f'Answers: {sample["answer"]}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n', file=f)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.', file=f)

    print(f"Final acc: {float(sum(answers))/len(answers)}", file=f)
    f.close()
    return