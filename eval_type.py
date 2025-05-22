import torch
import numpy as np
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitsProcessorList
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh

def log_likelihood(args, Model, question, answer, best_layer):
    with torch.no_grad():
        question = str(question)
        answer = str(answer)
        # import pdb; pdb.set_trace()
        if answer[0] != ' ':
            input_text = question + ' ' + answer
        else:
            input_text = question + answer

        input_ids = Model.tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        question_ids = Model.tokenizer(question, return_tensors="pt").input_ids.to('cuda')
        answer_ids = input_ids[0, question_ids.shape[-1]:] # tokenize(answer)

        outputs = Model.model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
        
        logits = outputs.hidden_states
        
        mature_layer = len(logits)-1
        head_layer = Model.model.get_output_embeddings()

        final_logits = head_layer(logits[mature_layer])[0, question_ids.shape[-1] - 1: -1, :]
        final_logits = final_logits.log_softmax(dim=-1)

        # stacked = [final_logits[0, answer_ids]]
        # for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
        #     base_logits = head_layer(logits[i])[0, question_ids.shape[-1] - 1: -1, :]
        #     base_logits = base_logits.log_softmax(dim=-1)
        #     stacked.append(base_logits[0, answer_ids])

        # stacked_tensors = torch.stack(stacked)
        # HR_5 = stacked_tensors.cpu().numpy()
        # sns.set(font_scale=1.5)
        # sns.set_context({"figure.figsize":(48,15)})
        # heatmap = sns.heatmap(HR_5, annot=True, linewidths=1.5, fmt=".2f", cbar=False,
        #                     linecolor="black", cmap="summer", 
        #                     yticklabels=[32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0],
        #                     xticklabels=[i for i in range(1, HR_5.shape[1]+1)])
        # heatmap.xaxis.tick_top()
        # plt.yticks(rotation=0)
        # plt.ylabel('i-th early exit layer')
        # plt.savefig('heat.svg', format="svg")
        # # plt.savefig('heat.jpg')
        # import pdb; pdb.set_trace()
        # plt.clf()
        # # plt.savefig('heat.svg', format="svg")
        
        # best_layer = 1
        # best_layer = random.randint(0, args.num_labels-1)
        if best_layer == -1:
            relative_top_mask = get_relative_top_filter(final_logits, 0.1)
            final_logits = torch.where(relative_top_mask, -1000, final_logits)
            device = final_logits.device
            answer_ids = answer_ids.to(device)
            log_prob = final_logits[range(final_logits.shape[0]), answer_ids].sum().item()
            return log_prob

        premature_layer = best_layer
        base_logits = head_layer(logits[premature_layer])[0, question_ids.shape[-1] - 1: -1, :]
        base_logits = base_logits.log_softmax(dim=-1)

        diff_logits = final_logits - base_logits
        diff_logits = diff_logits.log_softmax(dim=-1)

        relative_top_mask = get_relative_top_filter(final_logits, 0.1)
        diff_logits = torch.where(relative_top_mask, -1000, diff_logits)

        device = diff_logits.device
        answer_ids = answer_ids.to(device)

        log_prob = diff_logits[range(diff_logits.shape[0]), answer_ids].sum().item()
    
    return log_prob

def base_log(args, Model, question, answer, useless):
    question = str(question)
    answer = str(answer)

    if answer[0] != ' ':
        input_text = question + ' ' + answer
    else:
        input_text = question + answer

    input_ids = Model.tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    question_ids = Model.tokenizer(question, return_tensors="pt").input_ids.to('cuda')
    answer_ids = input_ids[0, question_ids.shape[-1]:] # tokenize(answer)

    with torch.no_grad():
        outputs = Model.model(input_ids)[0].squeeze(0)
        outputs = outputs.log_softmax(-1)  # logits to log probs

        outputs = outputs[question_ids.shape[-1] - 1: -1, :]
        log_probs = outputs[range(outputs.shape[0]), answer_ids].sum().item()

    return log_probs

def base_gen(args, Model, shot, question, useless1):
    if args.dataset == 'Gsm8k':
        stop_word_list = ["Q:", "\end{code}"]
    elif args.dataset == 'StraQA':
        stop_word_list = ["Q:", "\n\n##"]
    elif args.dataset == 'Folio':
        stop_word_list = ["Context:", "\n\n##"]
    elif args.dataset == 'AIME':
        stop_word_list = ["Q:", "\end{code}"]
    stopping_words = get_stop_wordids(Model.tokenizer, stop_word_list)

    input_text = shot + question
    inputs = Model.tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to('cuda')
    input_ids_all = input_ids

    attention_mask = inputs.attention_mask.to('cuda')

    past_key_values = None
    new_tokens_list = []
    max_new_tokens = 512

    with torch.no_grad():
        # logits & criteria processor
        processors = LogitsProcessorList()
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))

        for _ in range(max_new_tokens):
            outputs = Model.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True,
                            past_key_values=past_key_values
                        )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:,-1,:]
            next_token_logits = logits.log_softmax(dim=-1)

            next_token_logits = processors(input_ids_all, next_token_logits)
            next_token = torch.argmax(next_token_logits, dim=-1)

            new_tokens_list.append(next_token.item())
            if next_token.item() == Model.tokenizer.eos_token_id:
                break

            new_tokens_list, stop = if_stop(new_tokens_list, stopping_words)
            if stop:
                break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)

        output_sequence = Model.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
    return output_sequence, {i:0 for i in range(17)}

def generate(args, Model, shot, question, Classifier):
    if args.dataset == 'Gsm8k':
        stop_word_list = ["Q:", "\end{code}"]
    elif args.dataset == 'StraQA':
        stop_word_list = ["Q:", "\n\n##"]
    elif args.dataset == 'Folio':
        stop_word_list = ["Context:", "\n\n##"]
    stopping_words = get_stop_wordids(Model.tokenizer, stop_word_list)
    # import pdb; pdb.set_trace()
    input_text = shot + question
    inputs = Model.tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to('cuda')
    input_ids_all = input_ids
    attention_mask = inputs.attention_mask.to('cuda')
    past_key_values = None
    new_tokens_list = []

    shot_ids = Model.tokenizer(shot, return_tensors="pt").input_ids.to('cuda')
    question_ids = input_ids[0, shot_ids.shape[-1]:]

    best_layer_dic = {i:0 for i in range(17)}
    best_layer_dic[-1] = 0
    
    with torch.no_grad():
        processors = LogitsProcessorList()
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))

        max_new_tokens = 512
        for _ in range(max_new_tokens):
            outputs = Model.model(input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.hidden_states
            past_key_values = outputs.past_key_values
            mature_layer = len(logits)-1
            
            head_layer = Model.model.get_output_embeddings()
            final_logits = head_layer(logits[mature_layer])[:, -1, :]
            final_logits = final_logits.log_softmax(dim=-1)
            
            best_layer = Classifier.predict(Model.model, Model.tokenizer.decode(question_ids))
            # best_layer = 0
            # best_layer = random.randint(0, args.num_labels-1)
            if best_layer == -1:
                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                mask = final_logits[0] < -1e3
                final_logits[0][mask] = -1e3

                next_token_logits = final_logits
            
            else:
                # import pdb; pdb.set_trace()
                base_logits = head_layer(logits[best_layer])[:, -1, :]
                base_logits = base_logits.log_softmax(dim=-1)

                relative_top_mask = get_relative_top_filter(final_logits, 0.1)
                final_logits = torch.where(relative_top_mask, -1000, final_logits)
                # # ？无效操作
                # mask = final_logits[0] < -1e3
                # base_logits[0][mask] = -1e3

                next_token_logits = final_logits - base_logits

            input_ids_all = input_ids_all.to(next_token_logits.device)
            next_token_logits = processors(input_ids_all, next_token_logits)
            next_token = torch.argmax(next_token_logits, dim=-1)

            new_tokens_list.append(next_token.item())
            if next_token.item() == Model.tokenizer.eos_token_id:
                break

            new_tokens_list, stop = if_stop(new_tokens_list, stopping_words)
            if stop:
                break

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            input_ids_all = torch.cat([input_ids_all, next_token[:, None]], dim=-1)
            # import pdb; pdb.set_trace()
            question_ids = question_ids.to(next_token.device)
            question_ids = torch.cat([question_ids, next_token[0, None]], dim=-1)
            best_layer_dic[best_layer] += 1
        output_sequence = Model.tokenizer.decode(new_tokens_list, skip_special_tokens=True)
        return output_sequence, best_layer_dic
    

# special
def get_stop_wordids(tokenizer, stop_words):
    list_stop_word_ids = []
    for stop_word in stop_words:
        stop_word_ids = tokenizer.encode('\n' + stop_word)[2:]
        list_stop_word_ids.append(stop_word_ids)
        # print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    return list_stop_word_ids

def if_stop(new_tokens_list, stopping_words):
    end_words = [i[-1] for i in stopping_words]
    new_token = new_tokens_list[-1]
    if new_token in end_words:
        idx = end_words.index(new_token)
        seq_len = len(new_tokens_list)
        word_len = len(stopping_words[idx])
        if new_tokens_list[seq_len-word_len:] == stopping_words[idx]:
            return new_tokens_list[:seq_len-word_len], True
        else:
            return new_tokens_list, False
    else:
        return new_tokens_list, False