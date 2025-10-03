# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import torch
import torch.nn as nn
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from peft import LoraConfig, PeftModel, get_peft_model
import os
import src.dist_utils as dist_utils
import random
import torch.nn.functional as F
from transformers import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "last_hidden_states", "labels"])
PredLengthOutputs = namedtuple("PredLengthOutputs", ["loss", "inputs_embeds", "last_hidden_states", "labels", "ntp_loss"])

class HungarianMSELoss(nn.Module):
    """
    Computes the Hungarian-matched MSE loss between two batches of sets of vectors.
    Args:
        input: Tensor of shape (batch_size, k, d)
        target: Tensor of shape (batch_size, k, d)
    Returns:
        Scalar tensor: average Hungarian-matched MSE loss over the batch
    """
    def __init__(self, force_match_first=False):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.force_match_first = force_match_first

    def compute_hungarian_loss(self, vector_1, vector_2):
        # Compute cost matrix of MSE between all pairs
        cost_matrix = torch.cdist(vector_1, vector_2, p=2).pow(2)  # (k, k)    
        row_ind, col_ind = self.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        # Gather matched pairs and compute MSE
        matched_input = vector_1[row_ind]
        matched_target = vector_2[col_ind]
        mse = torch.mean((matched_input - matched_target) ** 2)
        return mse

    def forward(self, input, target):
        # input, target: (batch_size, k, d)
        batch_size, k, d = input.shape
        losses = []
        for b in range(batch_size):
            if self.force_match_first:
                # If predicting question, force first embeddings to match
                mse = torch.mean((input[b][0] - target[b][0]) ** 2) + ((len(input[b])-1) * self.compute_hungarian_loss(input[b][1:], target[b][1:]))
                mse = mse / len(input[b])
            else:
                mse = self.compute_hungarian_loss(input[b], target[b])
            losses.append(mse)
        return torch.stack(losses).mean()
    

class ContrastiveLoss(nn.Module):
    """
    Computes the contrastive loss between two batches of sets of vectors.
    Args:
        input: Tensor of shape (batch_size, k, d)
        positive_embeddings: Tensor of shape (batch_size, k, d)
        negative_embeddings: Tensor of shape (batch_size, k, d)
    Returns:
        Scalar tensor: average contrastive loss over the batch
    """
    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)


    def forward(self, outputs, positive_embeddings, negative_embeddings):
        # normalize the embeddings
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)
        
        # input: (batch_size, k, d)
        # positive_embeddings: (batch_size, k, d)
        # negative_embeddings: (batch_size, k, d) 
                   
        batch_size, k, d = outputs.shape
        labels = torch.arange(batch_size * k).long().to(outputs.device)
        outputs = outputs.view(batch_size * k, d)  # (batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)  # (batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)  # (batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)  # (2 * batch_size * k, d)
        
        
        # handle distributed training
        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
        labels = labels + start_idx
        gather_fn = dist_utils.varsize_gather
        gather_kemb = gather_fn(all_embeddings)
        
        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)  # (batch_size * k, 2 * batch_size * k * num_gpus)
        loss = self.ce_loss(similarity, labels)
        loss = loss.mean()
        return loss
    
class HungarianContrastiveLoss(nn.Module):
    """
    Computes the contrastive loss between two batches of sets of vectors.
    Args:
        input: Tensor of shape (batch_size, k, d)
        positive_embeddings: Tensor of shape (batch_size, k, d)
        negative_embeddings: Tensor of shape (batch_size, k, d)
    Returns:
        Scalar tensor: average contrastive loss over the batch
    """
    def __init__(self, temperature=0.05, use_eos=False, normalize_embeddings=True):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.use_eos = use_eos
        self.normalize_embeddings = normalize_embeddings
            
    def forward(self, outputs, positive_embeddings, negative_embeddings):
        # normalize the embeddings
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        # input: (batch_size, k, d)
        # positive_embeddings: (batch_size, k, d)
        # negative_embeddings: (batch_size, k, d)
        batch_size, k, d = outputs.shape
        # labels = torch.arange(batch_size * k).long().to(outputs.device)
        outputs = outputs.view(batch_size * k, d)  # (batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)  # (batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)  # (batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)  # (2 * batch_size * k, d)
        var_sizes = dist_utils.get_varsize(all_embeddings)
        
        # handle distributed training
        gather_fn = dist_utils.varsize_gather
        gather_kemb = gather_fn(all_embeddings)
        
        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)  # (batch_size * k, 2 * batch_size * k * num_gpus)
        similarity = self.log_softmax(similarity)
        losses = []
        for i in range(batch_size):
            # print('============================')
            start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
            start_idx += k*i 
            batch_scores = similarity[k*i:k*(i+1)]
            cost_matrix = batch_scores[:, start_idx:start_idx+k]
            
            if self.use_eos:
                # if use eos, force match the last token to the eos token
                cost_eos = cost_matrix[-1, -1]
                cost_matrix = cost_matrix[:-1, :-1]
            
            row_ind, col_ind = self.linear_sum_assignment(cost_matrix.detach().cpu().numpy(), maximize=True)
            costs = cost_matrix[row_ind, col_ind]  # (k, )
            if self.use_eos:
                costs = torch.cat([costs, cost_eos.unsqueeze(0)])
            costs = -(costs)
            losses.append(costs.mean())
        return torch.stack(losses).mean()


class EmbeddingModel(nn.Module):

    def __init__(
        self,
        base_causallm,
        start_latent_id,
        eos_token_id,
        embedding_model_dim,
        weight_tying=False,
        loss_function='Hungarian_MSE',
        temperature=0.05,
        extra_q_embed=False,
        compute_loss_on_q=False,
        use_eos=False,
        normalize_embeddings=True
    ):

        super(EmbeddingModel, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            
        hidden_size = self.base_causallm.config.hidden_size
        self.embedding_model_dim = embedding_model_dim
        self.weight_tying = weight_tying
        self.input_projection = nn.Linear(embedding_model_dim, hidden_size, bias=False).float()
        if weight_tying:
            self.output_projection = nn.Linear(hidden_size, embedding_model_dim, bias=False).float()
            self.output_projection.weight[:] = self.input_projection.weight.transpose(0, 1)[:]
        else:
            self.output_projection = nn.Linear(hidden_size, embedding_model_dim, bias=False).float()
            
        self.extra_q_embed = extra_q_embed
        self.compute_loss_on_q = compute_loss_on_q
        self.use_eos = use_eos
        self.normalize_embeddings = normalize_embeddings

        if loss_function == 'Hungarian_MSE':
            self.loss_fct = HungarianMSELoss(force_match_first=self.extra_q_embed and self.compute_loss_on_q)
        elif loss_function == 'Contrastive':
            self.loss_fct = ContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        elif loss_function == 'Hungarian_Contrastive':
            self.loss_fct = HungarianContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        elif loss_function == 'MSE':
            self.loss_fct = torch.nn.MSELoss()
        else:
            raise ValueError(f"Loss function {loss_function} not supported")
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, **inputs):
        has_label = 'labels' in inputs or 'positive_embeddings' in inputs
        if has_label:  # the labels could be either used for MSE loss or contrastive loss
            if 'labels' in inputs:
                labels = inputs.pop("labels")
                label_type = 'labels'
            else:
                labels = inputs.pop("positive_embeddings")
                label_type = 'positive_embeddings'
        
        assert has_label, "only support training now"
        loss_mask = inputs['attention_mask'].detach().clone()
        
        # get the input embeddings from the base causal language model
        if 'input_ids' in inputs:
            outputs = self.base_causallm(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        else:
            outputs = self.base_causallm(inputs_embeds=self.input_projection(inputs['inputs_embeds']), attention_mask=inputs['attention_mask'], output_hidden_states=True)
        inputs['hidden_states'] = outputs.hidden_states[0].clone().detach()
        
        for i in range(inputs['hidden_states'].size(0)):
            # assign the labels to the hidden states as input
            input_start_for_output = inputs['attention_mask'][i].sum()
            
            # [1, 257, 2048], [1, 3, 2048], [1, 3, 1536]
            output_len = labels[i].size(0)
            inputs['hidden_states'][i][input_start_for_output:input_start_for_output+output_len,:] = self.input_projection(labels[i].float())
            
            # ignore the first token, which is the question representation using embedding model
            if self.extra_q_embed and not self.compute_loss_on_q:  # only when we have extra question embeddings and we don't compute loss on the question embeddings
                loss_mask[i,:input_start_for_output] = 0
            else:
                loss_mask[i,:input_start_for_output-1] = 0                

            # fill out the loss mask and attention mask
            if self.extra_q_embed and not self.compute_loss_on_q:
                loss_mask[i,input_start_for_output:(input_start_for_output+output_len-1)] = 1
                assert loss_mask[i].sum().item() == (output_len - 1), (loss_mask[i].sum().item(), output_len)
            else:
                loss_mask[i,(input_start_for_output-1):(input_start_for_output+output_len-1)] = 1
                assert loss_mask[i].sum().item() == (output_len), (loss_mask[i].sum().item(), output_len)
            
            inputs['attention_mask'][i,input_start_for_output:(input_start_for_output+output_len)] = 1
            assert inputs['attention_mask'][i].sum().item() == (input_start_for_output + output_len)
        
        inputs['inputs_embeds'] = inputs['hidden_states']
        del inputs['hidden_states']
        
        outputs = self.base_causallm(inputs_embeds=inputs['inputs_embeds'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        out_hidden_states = outputs.hidden_states[-1]

        if has_label:
            if self.extra_q_embed and not self.compute_loss_on_q:
                labels = labels[:,1:,:] # only takes the outputs tokens, ignoring the first token (which is the question representation using embedding model)
            # Get indices where loss_mask is 1
            if len(loss_mask.nonzero().size()) > 2:
                mask_indices = loss_mask.nonzero().squeeze()
            else:
                mask_indices = loss_mask.nonzero()
            selected_out_hidden_states = out_hidden_states[mask_indices[:, 0], mask_indices[:, 1]]
            # Select only the hidden states where mask is 1
            selected_outputs_embeddings = self.output_projection(selected_out_hidden_states).contiguous()
            selected_outputs_embeddings = selected_outputs_embeddings.view(labels.size(0), labels.size(1), -1)  # (batch_size, length, embedding_dim)
            assert selected_outputs_embeddings.size() == labels.size(), (selected_outputs_embeddings.size(), labels.size())
            
            if label_type == 'labels':
                #########################################################
                # MSE loss
                #########################################################
                loss = self.loss_fct(selected_outputs_embeddings, labels.float())
                return Outputs(loss=loss, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=labels)
            elif label_type == 'positive_embeddings':
                #########################################################
                # Contrastive loss
                #########################################################
                positive_embeddings = labels
                negative_embeddings = inputs.pop("negative_embeddings")
                if self.use_eos:
                    loss = self.loss_fct(selected_outputs_embeddings[:, :-1, :], positive_embeddings[:, :-1, :], negative_embeddings[:, :-1, :])
                    loss += ((selected_outputs_embeddings[:, -1, :] - 0.5)**2).mean()
                    
                loss = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
                return Outputs(loss=loss, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=labels)
            else:
                raise ValueError("No positive embeddings found")
                
        else:
            return Outputs(loss=None, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=None)
        
        

    def generate(
        self,
        max_new_tokens=16, 
        use_gt_q_embed=False,
        use_eos=False,
        **inputs
    ):
        self.gen_forward_cnt = 0
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        elif 'inputs_embeds' in inputs:
            inputs_embeds = inputs['inputs_embeds']
            assert inputs_embeds.shape[0] == 1, "only support batch_size == 1 now"
        else:
            hidden_states = inputs['hidden_states']
            assert hidden_states.shape[0] == 1, "only support batch_size == 1 now"
        
        next_embs = []
        
        assert 'input_ids' in inputs or 'inputs_embeds' in inputs, "only support input_ids or inputs_embeds now"
        if 'input_ids' in inputs:
            assert inputs['input_ids'].size(1) == inputs['attention_mask'].sum(), (inputs['input_ids'].size(1), inputs['attention_mask'].sum())
        else:
            assert inputs['inputs_embeds'].size(1) == inputs['attention_mask'].sum(), (inputs['inputs_embeds'].size(1), inputs['attention_mask'].sum())
        
        # predict the first pass; also get the input embeddings from the base causal language model
        if 'input_ids' in inputs:
            outputs = self.base_causallm(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        else:
            outputs = self.base_causallm(inputs_embeds=self.input_projection(inputs['inputs_embeds'].float()), output_hidden_states=True)
        inputs['hidden_states'] = outputs.hidden_states[0]
        
        
        if use_gt_q_embed: # use the ground truth question embeddings; the first step doesn't count, generate the rest of the tokens
            question_embeddings = self.input_projection(inputs['question_embeddings'])  
            new_inputs_embeds = torch.cat((inputs['hidden_states'], question_embeddings.unsqueeze(1)), dim=1)
        else:              # do not use the ground truth question embeddings; the first step counts, generate the rest of the tokens
            if max_new_tokens == 1:  # only predict the question embeddings
                out_embs = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                return self.output_projection(out_embs)
            
            next_emb = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            next_embs.append(next_emb)
            new_inputs_embeds = torch.cat((inputs['hidden_states'], self.input_projection(self.output_projection(next_emb))), dim=1)  
            max_new_tokens = max_new_tokens - 1

        # generate the rest of the tokens
        for _ in range(max_new_tokens):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds, output_hidden_states=True)
            self.gen_forward_cnt += 1
            next_emb = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if use_eos:
                print("next_emb", next_emb.shape, (next_emb - 0.5).abs().mean(), next_emb)
                if (next_emb - 0.5).abs().mean() < 1e-4:
                    print("EOS token generated")
                    break
            next_embs.append(next_emb)
            new_inputs_embeds = torch.cat((new_inputs_embeds, self.input_projection(self.output_projection(next_emb))), dim=1)
        
        out_embs = torch.cat(next_embs, dim=1)
        
        out_embs = self.output_projection(out_embs)
        return out_embs
      

class EmbeddingModelSS(EmbeddingModel):

    def __init__(
        self,
        base_causallm,
        start_latent_id,
        eos_token_id,
        embedding_model_dim,
        weight_tying=False,
        loss_function='Hungarian_MSE',
        temperature=0.05,
        extra_q_embed=False,
        compute_loss_on_q=False,
        use_eos=False,
        normalize_embeddings=True
    ):
        super(EmbeddingModelSS, self).__init__(
            base_causallm=base_causallm,
            start_latent_id=start_latent_id,
            eos_token_id=eos_token_id,
            embedding_model_dim=embedding_model_dim,
            weight_tying=weight_tying,
            loss_function=loss_function,
            temperature=temperature,
            extra_q_embed=extra_q_embed,
            compute_loss_on_q=compute_loss_on_q,
            use_eos=use_eos,
            normalize_embeddings=normalize_embeddings
        )

    def forward(self, **inputs):
        has_label = 'labels' in inputs or 'positive_embeddings' in inputs
        if has_label:  # the labels could be either used for MSE loss or contrastive loss
            if 'labels' in inputs:
                labels = inputs.pop("labels")
                label_type = 'labels'
            else:
                labels = inputs.pop("positive_embeddings")
                label_type = 'positive_embeddings'
        
        assert has_label, "only support training now"
        loss_mask = inputs['attention_mask'].detach().clone()
        
        # get the input embeddings from the base causal language model
        if 'input_ids' in inputs:
            outputs = self.base_causallm(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        else:
            outputs = self.base_causallm(inputs_embeds=self.input_projection(inputs['inputs_embeds']), attention_mask=inputs['attention_mask'], output_hidden_states=True)
        inputs['hidden_states'] = outputs.hidden_states[0].clone().detach()
        
        # assign the labels to the hidden states as input
        inputs['inputs_embeds'] = inputs['hidden_states']
        del inputs['hidden_states']
        
        ## Use Generation Instead of Forward Pass
        sampling_rate = inputs['sampling_rate']
        input_start_for_output = inputs['attention_mask'][0].sum()
        for i in range(inputs['attention_mask'].size(0)):
            assert inputs['attention_mask'][i].sum() == input_start_for_output, (inputs['attention_mask'][i].sum(), input_start_for_output)
        output_len = labels[0].size(0)
        for i in range(labels.size(0)):
            assert labels[i].size(0) == output_len, (labels[i].size(0), output_len)
        
        out_hidden_states = []    
        current_input = inputs['inputs_embeds'][:, :input_start_for_output] # (1, input_start_for_output, hidden_size)
        
        # Generate the output tokens
        all_outputs = []
        for j in range(output_len):
            # Generate the next token
            outputs = self.base_causallm(inputs_embeds=current_input, output_hidden_states=True)
            next_emb = outputs.hidden_states[-1][:, input_start_for_output+j-1, :] # (batch_size, hidden_size)
            all_outputs.append(next_emb.unsqueeze(1))
            
            # do the sampling 
            use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)
            predicted = self.input_projection(self.output_projection(next_emb)).unsqueeze(1)
            teacher = self.input_projection(labels[:, j].float()).unsqueeze(1)
            next_input = torch.where(use_predicted, predicted, teacher)
            current_input = torch.cat((current_input, next_input), dim=1)
        
        if self.extra_q_embed and not self.compute_loss_on_q:  # only when we have extra question embeddings and we don't compute loss on the question embeddings
            loss_mask[:,:1] = 0
            loss_mask[:,1:output_len] = 1
            assert loss_mask.float().mean(dim=0).sum().item() == (output_len - 1), (loss_mask.float().mean(dim=0).sum().item(), output_len)
        else:
            loss_mask[:,:output_len] = 1
            assert loss_mask.float().mean(dim=0).sum().item() == (output_len), (loss_mask.float().mean(dim=0).sum().item(), output_len)
                
        out_hidden_states = torch.cat(all_outputs, dim=1) # (batch_size, output_len, hidden_size)

        if has_label:
            if self.extra_q_embed and not self.compute_loss_on_q:
                labels = labels[:,1:,:] # only takes the outputs tokens, ignoring the first token (which is the question representation using embedding model)
            # Get indices where loss_mask is 1
            if len(loss_mask.nonzero().size()) > 2:
                mask_indices = loss_mask.nonzero().squeeze()
            else:
                mask_indices = loss_mask.nonzero()
            selected_out_hidden_states = out_hidden_states[mask_indices[:, 0], mask_indices[:, 1]]
            # Select only the hidden states where mask is 1
            selected_outputs_embeddings = self.output_projection(selected_out_hidden_states).contiguous()
            selected_outputs_embeddings = selected_outputs_embeddings.view(labels.size(0), labels.size(1), -1)  # (batch_size, length, embedding_dim)
            assert selected_outputs_embeddings.size() == labels.size(), (selected_outputs_embeddings.size(), labels.size())
            
            if label_type == 'labels':
                #########################################################
                # MSE loss
                #########################################################
                loss = self.loss_fct(selected_outputs_embeddings, labels.float())
                return Outputs(loss=loss, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=labels)
            elif label_type == 'positive_embeddings':
                #########################################################
                # Contrastive loss
                #########################################################
                positive_embeddings = labels
                negative_embeddings = inputs.pop("negative_embeddings")
                if self.use_eos:
                    loss = self.loss_fct(selected_outputs_embeddings[:, :-1, :], positive_embeddings[:, :-1, :], negative_embeddings[:, :-1, :])
                    loss += ((selected_outputs_embeddings[:, -1, :] - 0.5)**2).mean()
                    
                loss = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
                return Outputs(loss=loss, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=labels)
            else:
                raise ValueError("No positive embeddings found")
                
        else:
            return Outputs(loss=None, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=selected_outputs_embeddings, labels=None)
        

        
class EmbeddingModelSSVariableLeftPad(EmbeddingModel):

    def __init__(
        self,
        base_causallm,
        start_latent_id,
        eos_token_id,
        embedding_model_dim,
        weight_tying=False,
        loss_function='Hungarian_MSE',
        temperature=0.05,
        extra_q_embed=False,
        compute_loss_on_q=False,
        use_eos=False,
        normalize_embeddings=True
    ):
        super(EmbeddingModelSSVariableLeftPad, self).__init__(
            base_causallm=base_causallm,
            start_latent_id=start_latent_id,
            eos_token_id=eos_token_id,
            embedding_model_dim=embedding_model_dim,
            weight_tying=weight_tying,
            loss_function=loss_function,
            temperature=temperature,
            extra_q_embed=extra_q_embed,
            compute_loss_on_q=compute_loss_on_q,
            use_eos=use_eos,
            normalize_embeddings=normalize_embeddings
        )

    def forward(self, **inputs):
        has_label = 'labels' in inputs or 'positive_embeddings' in inputs
        if has_label:  # the labels could be either used for MSE loss or contrastive loss
            if 'labels' in inputs:
                labels = inputs.pop("labels")
                label_type = 'labels'
            else:
                labels = inputs.pop("positive_embeddings")
                label_type = 'positive_embeddings'
        
        assert has_label, "only support training now"
        loss_mask = inputs['attention_mask'].detach().clone()
        assert 'position_ids' in inputs, "position_ids is required"
        output_len = labels[0].size(0)
        for i in range(labels.size(0)):
            assert labels[i].size(0) == output_len, (labels[i].size(0), output_len)

        # get the input embeddings from the base causal language model
        if 'input_ids' in inputs:
            outputs = self.base_causallm(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], position_ids=inputs['position_ids'], output_hidden_states=True)
        else:
            outputs = self.base_causallm(inputs_embeds=self.input_projection(inputs['inputs_embeds']), attention_mask=inputs['attention_mask'], position_ids=inputs['position_ids'], output_hidden_states=True)
        input_start_for_output = inputs['attention_mask'].sum(dim=1).max()
        assert inputs['attention_mask'][:, input_start_for_output:].sum().item() == 0, (inputs['attention_mask'][:, input_start_for_output:].sum().item(), input_start_for_output)
        
        current_input = outputs.hidden_states[0][:, :input_start_for_output, :]
        
        ## Use Generation Instead of Forward Pass
        sampling_rate = inputs['sampling_rate']

        out_hidden_states = []
        all_outputs = []
        for j in range(output_len):
            # Generate the next token
            outputs = self.base_causallm(inputs_embeds=current_input, position_ids=inputs['position_ids'][:, :input_start_for_output+j], attention_mask=inputs['attention_mask'][:, :input_start_for_output+j], output_hidden_states=True)
            
            next_emb = outputs.hidden_states[-1][:, input_start_for_output+j-1, :] # (batch_size, hidden_size)
            all_outputs.append(next_emb.unsqueeze(1))                         # (batch_size, 1, hidden_size)
            
            # do the sampling 
            use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)                 # (batch_size, 1)
            predicted = self.input_projection(self.output_projection(next_emb)).unsqueeze(1)                                # (batch_size, 1, hidden_size)
            teacher = self.input_projection(labels[:, j].float()).unsqueeze(1)
            next_input = torch.where(use_predicted, predicted, teacher)
            current_input = torch.cat((current_input, next_input), dim=1)
  
            ## update the position ids & attention mask
            inputs['position_ids'][:, input_start_for_output+j:] = inputs['position_ids'][:, input_start_for_output+j:] + 1
            inputs['attention_mask'][:, input_start_for_output+j] = 1
            
        # fill out the loss mask: ignore the first token, which is the question representation using embedding model
        if self.extra_q_embed and not self.compute_loss_on_q:  # only when we have extra question embeddings and we don't compute loss on the question embeddings
            loss_mask[:,:1] = 0
            loss_mask[:,1:output_len] = 1
            loss_mask[:,output_len:] = 0
            assert loss_mask.float().mean(dim=0).sum().item() == (output_len - 1), (loss_mask.float().mean(dim=0).sum().item(), output_len)
        else:
            loss_mask[:,:output_len] = 1
            loss_mask[:,output_len:] = 0
            assert loss_mask.float().mean(dim=0).sum().item() == (output_len), (loss_mask.float().mean(dim=0).sum().item(), output_len)
        
        out_hidden_states = torch.cat(all_outputs, dim=1) # (batch_size, output_len, hidden_size)

        if has_label:
            if self.extra_q_embed and not self.compute_loss_on_q:
                labels = labels[:,1:,:] # only takes the outputs tokens, ignoring the first token (which is the question representation using embedding model)
            # Get indices where loss_mask is 1
            if len(loss_mask.nonzero().size()) > 2:
                mask_indices = loss_mask.nonzero().squeeze()
            else:
                mask_indices = loss_mask.nonzero()
            selected_out_hidden_states = out_hidden_states[mask_indices[:, 0], mask_indices[:, 1]]
            # Select only the hidden states where mask is 1
            selected_outputs_embeddings = self.output_projection(selected_out_hidden_states).contiguous()
            selected_outputs_embeddings = selected_outputs_embeddings.view(labels.size(0), labels.size(1), -1)  # (batch_size, length, embedding_dim)
            assert selected_outputs_embeddings.size() == labels.size(), (selected_outputs_embeddings.size(), labels.size())
            
            if label_type == 'labels':
                #########################################################
                # MSE loss
                #########################################################
                loss = self.loss_fct(selected_outputs_embeddings, labels.float())
                return Outputs(loss=loss, inputs_embeds=current_input, last_hidden_states=selected_outputs_embeddings, labels=labels)
            elif label_type == 'positive_embeddings':
                #########################################################
                # Contrastive loss
                #########################################################
                positive_embeddings = labels
                negative_embeddings = inputs.pop("negative_embeddings")
                if self.use_eos:
                    loss = self.loss_fct(selected_outputs_embeddings[:, :-1, :], positive_embeddings[:, :-1, :], negative_embeddings[:, :-1, :])
                    loss += ((selected_outputs_embeddings[:, -1, :] - 0.5)**2).mean()
                    
                loss = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
                return Outputs(loss=loss, inputs_embeds=current_input, last_hidden_states=selected_outputs_embeddings, labels=labels)
            else:
                raise ValueError("No positive embeddings found")
                
        else:
            return Outputs(loss=None, inputs_embeds=current_input, last_hidden_states=selected_outputs_embeddings, labels=None)
 

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(train_lora, base_model_id, adapter_path, linear_checkpoint_path, embedding_model_dim, 
               weight_tying=False, loss_function='Hungarian_MSE', temperature=0.05, lora_alpha=16, lora_r=64, lora_dropout=0.1, extra_q_embed=False, 
               compute_loss_on_q=False, use_eos=False, model_type="EmbeddingModel", normalize_embeddings=False):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.gradient_checkpointing_enable()
    
    # Load the tokenizer
    if 'full_finetuning' in base_model_id:
        if 'inf' in base_model_id.split('/')[1]:
            print('load tokenizer from the starting point, not from the finetuned model', 'infly/inf-retriever-v1-1.5b')
            tokenizer = AutoTokenizer.from_pretrained("infly/inf-retriever-v1-1.5b")
        elif 'llama' in base_model_id.split('/')[1]:
            print('load tokenizer from the starting point, not from the finetuned model', 'meta-llama/Llama-3.2-1B-Instruct')
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        else:
            raise ValueError(f"Invalid base model id: {base_model_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    start_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    
    if train_lora:
        if adapter_path is not None:
            # Load the model with LoRA adapter weights
            print('=======')
            print(f"loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        else:
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj".split(",")
            )
            model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
    else:
        if adapter_path is not None:
            # Load the model with LoRA adapter weights
            print('=======')
            print(f"loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
            model.print_trainable_parameters()
            for param in model.parameters():
                param.requires_grad = True
            print('setting all parameters to trainable')
            model.print_trainable_parameters()
        else:
            print("load base model")
            model = base_model
    
    # Wrap with your custom EmbeddingModel
    if model_type == "EmbeddingModel":
        model_class = EmbeddingModel
    elif model_type == "EmbeddingModelSS":
        model_class = EmbeddingModelSS
    elif model_type == "EmbeddingModelSSVariableLeftPad":
        model_class = EmbeddingModelSSVariableLeftPad
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    print(f"loading model {model_type}")
    model = model_class(model, start_id, tokenizer.eos_token_id, 
                           embedding_model_dim, weight_tying, loss_function, 
                           temperature, extra_q_embed, compute_loss_on_q, use_eos, normalize_embeddings)
    
    # Load linear layers if checkpoint is provided
    if linear_checkpoint_path is not None:
        print(f"loading linear layers from {linear_checkpoint_path}")
        linear_layers = torch.load(linear_checkpoint_path)
        model.input_projection.load_state_dict(linear_layers['input_projection'])
        model.output_projection.load_state_dict(linear_layers['output_projection'])
    
    return model, tokenizer


from collections import OrderedDict
def save_model_distributed(model, save_dir, step, eval_loss, accelerator, logger, save_best_model=False):
    # Save the base causal language model
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        linear_layers = {
            'input_projection': OrderedDict({'weight': state_dict['input_projection.weight']}),
            'output_projection': OrderedDict({'weight': state_dict['output_projection.weight']}),
            'step': step,
            'loss': eval_loss
        }
    else:
        linear_layers = None
    
    if save_best_model:
        base_save_dir = os.path.join(save_dir, f"best_model")
    else:
        base_save_dir = os.path.join(save_dir, f"checkpoint_{step}")
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.base_causallm.save_pretrained(
        base_save_dir, 
        safe_serialization=True,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict={
            k.replace("base_causallm.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("base_causallm.")
        }
    )
    
    # Save the linear layers
    if linear_layers is not None:
        if save_best_model:
            accelerator.save(linear_layers, os.path.join(save_dir, f"best_model_linear.pt"))
        else:
            accelerator.save(linear_layers, os.path.join(save_dir, f"checkpoint_{step}_linear.pt"))
        logger.info(f"saving model.", step=(step), process_index=accelerator.process_index)

    
    
def save_model_single(model, save_dir, step, eval_loss, logger, save_best_model=False):
    # Save the base causal language model
    if save_best_model:
        model.base_causallm.save_pretrained(os.path.join(save_dir, f"best_model"), safe_serialization=True)
    else:
        model.base_causallm.save_pretrained(os.path.join(save_dir, f"checkpoint_{step}"), safe_serialization=True)
    
    # Save the linear layers
    linear_layers = {
        'input_projection': model.input_projection.state_dict(),
        'output_projection': model.output_projection.state_dict(),
        'step': step,
        'loss': eval_loss
    }
    if save_best_model:
        torch.save(linear_layers, os.path.join(save_dir, f"best_model_linear.pt"))
    else:
        torch.save(linear_layers, os.path.join(save_dir, f"checkpoint_{step}_linear.pt"))
    logger.info(f"saving model.", step=(step))