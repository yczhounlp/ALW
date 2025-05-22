import torch
import numpy as np
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitsProcessorList

def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized < probs_thresh

def log_likelihood(args, Model, question, answer):
    log_probs = []
    early_exit_layers = [i for i in range(args.layers+1)]
    with torch.no_grad():
        question = str(question)
        answer = str(answer)
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
        # print(f"Modle have {mature_layer} layers.")

        early_exit_layers.append(mature_layer)
        dict_outputs = {int(layer):0 for layer in early_exit_layers}
        head_layer = Model.model.get_output_embeddings()
        for l in dict_outputs.keys():
            dict_outputs[l] = head_layer(logits[l])
        
        final_logits = dict_outputs[mature_layer][0, question_ids.shape[-1] - 1: -1, :]
        final_logits = final_logits.log_softmax(dim=-1)

        for premature_layer in early_exit_layers[:-1]:

            base_logits = dict_outputs[premature_layer][0, question_ids.shape[-1] - 1: -1, :]
            base_logits = base_logits.log_softmax(dim=-1)

            diff_logits = final_logits - base_logits
            diff_logits = diff_logits.log_softmax(dim=-1) # post softmax

            relative_top_mask = get_relative_top_filter(final_logits, 0.1)
            diff_logits = torch.where(relative_top_mask, -1000, diff_logits)

            device = diff_logits.device
            answer_ids = answer_ids.to(device)

            log_prob = diff_logits[range(diff_logits.shape[0]), answer_ids].sum().item()
            log_probs.append(log_prob)
        
        # 考虑final layer是否可以做对,放在log_probs最后一个位置
        relative_top_mask = get_relative_top_filter(final_logits, 0.1)
        final_logits = torch.where(relative_top_mask, -1000, final_logits)
        log_prob = final_logits[range(final_logits.shape[0]), answer_ids].sum().item()
        log_probs.append(log_prob)

        return log_probs

def generate(args, Model, shot, question, answer):
    early_exit_layers = [i for i in range(args.layers+1)]
    input_text = shot + question + answer
    input_ids = Model.tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    shotques_ids = Model.tokenizer(shot + question, return_tensors="pt").input_ids.to('cuda')
    shot_ids = Model.tokenizer(shot, return_tensors="pt").input_ids.to('cuda')
    answer_ids = input_ids[0, shotques_ids.shape[-1]:]
    question_ids = input_ids[0, shot_ids.shape[-1]:-answer_ids.shape[-1]]

    with torch.no_grad():
        processors = LogitsProcessorList()
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=1.2))

        outputs = Model.model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
        
        logits = outputs.hidden_states
        mature_layer = len(logits)-1

        early_exit_layers.append(mature_layer)
        dict_outputs = {int(layer):0 for layer in early_exit_layers}
        head_layer = Model.model.get_output_embeddings()
        for l in dict_outputs.keys():
            dict_outputs[l] = head_layer(logits[l])
        
        final_logits = dict_outputs[mature_layer][0, shotques_ids.shape[-1] - 1: -1, :]
        final_logits = final_logits.log_softmax(dim=-1)

        layer_probs = []
        for premature_layer in early_exit_layers[:-1]:
            probs = []
            base_logits = dict_outputs[premature_layer][0, shotques_ids.shape[-1] - 1: -1, :]
            base_logits = base_logits.log_softmax(dim=-1)

            relative_top_mask = get_relative_top_filter(final_logits, 0.1)
            final_logits = torch.where(relative_top_mask, -1000, final_logits)
            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            diff_logits = final_logits - base_logits
            start = shotques_ids.shape[-1]-1
            for current in range(answer_ids.shape[-1]):
                device = diff_logits.device
                input_ids_all = input_ids[0, :start+current+1].unsqueeze(0).to(device)
                current_logits = processors(input_ids_all, diff_logits[current].unsqueeze(0))
                current_logits = current_logits.softmax(dim=-1)
                probs.append(current_logits[0, input_ids[0, start+current+1]].item())

            layer_probs.append(probs)

        # 考虑final layer
        probs = []
        relative_top_mask = get_relative_top_filter(final_logits, 0.1)
        final_logits = torch.where(relative_top_mask, -1000, final_logits)
        mask = final_logits[0] < -1e3
        final_logits[0][mask] = -1e3

        for current in range(answer_ids.shape[-1]):
            device = final_logits.device
            input_ids_all = input_ids[0, :start+current+1].unsqueeze(0).to(device)
            
            current_logits = processors(input_ids_all, final_logits[current].unsqueeze(0))
            current_logits = current_logits.softmax(dim=-1)
            probs.append(current_logits[0, input_ids[0, start+current+1]].item())
        
        layer_probs.append(probs)

        # find best layer
        layer_probs = np.array(layer_probs)
        best_layers = np.argmax(layer_probs, axis=0)
        max_values = layer_probs[best_layers, np.arange(best_layers.shape[0])]
        best_layers = np.where(max_values == layer_probs[-1, :], -1, best_layers)

        context = [Model.tokenizer.decode(question_ids)]
        for i in range(len(best_layers)-1):
            context.append(Model.tokenizer.decode(
                        torch.cat((question_ids, answer_ids[:i+1]))
                        ))

        assert len(context) == len(best_layers)
        return context, best_layers.tolist()