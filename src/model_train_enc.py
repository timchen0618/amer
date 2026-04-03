# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import torch
import torch.nn as nn
from collections import namedtuple, OrderedDict
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, PeftModel, get_peft_model
import os
import src.dist_utils as dist_utils
import torch.nn.functional as F

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "last_hidden_states", "labels"])


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, outputs, positive_embeddings, negative_embeddings):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        labels = torch.arange(batch_size * k).long().to(outputs.device)
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
        labels = labels + start_idx
        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        loss = self.ce_loss(similarity, labels)
        return loss.mean()


class HungarianContrastiveLoss(nn.Module):
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
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)

        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        similarity = self.log_softmax(similarity)
        losses = []
        for i in range(batch_size):
            start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
            start_idx += k * i
            batch_scores = similarity[k * i:k * (i + 1)]
            cost_matrix = batch_scores[:, start_idx:start_idx + k]

            if self.use_eos:
                cost_eos = cost_matrix[-1, -1]
                cost_matrix = cost_matrix[:-1, :-1]

            row_ind, col_ind = self.linear_sum_assignment(cost_matrix.detach().cpu().numpy(), maximize=True)
            costs = cost_matrix[row_ind, col_ind]
            if self.use_eos:
                costs = torch.cat([costs, cost_eos.unsqueeze(0)])
            costs = -(costs)
            losses.append(costs.mean())
        return torch.stack(losses).mean()


class EmbeddingModelDocEnc(nn.Module):
    """
    Autoregressive multi-query embedding model with a shared encoder for both
    queries and documents.

    The same base_causallm is used to:
    1. Encode documents via last-token pooling + output_projection → document embeddings
    2. Autoregressively generate multiple query embeddings

    Expected batch keys (training):
        - input_ids: (batch_size, seq_len) — tokenized query
        - attention_mask: (batch_size, seq_len)
        - position_ids: (batch_size, seq_len)
        - input_document_ids: (batch_size, num_documents, doc_seq_len)
            First half are positive docs, second half are negative docs.
        - attention_mask_document: (batch_size, num_documents, doc_seq_len)
        - sampling_rate: float — scheduled sampling probability
    """

    def __init__(
        self,
        base_causallm,
        start_latent_id,
        eos_token_id,
        embedding_model_dim,
        weight_tying=False,
        loss_function='Hungarian_Contrastive',
        temperature=0.05,
        normalize_embeddings=True
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id

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

        self.normalize_embeddings = normalize_embeddings

        if loss_function == 'Contrastive':
            self.loss_fct = ContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        elif loss_function == 'Hungarian_Contrastive':
            self.loss_fct = HungarianContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        else:
            raise ValueError(f"Loss function {loss_function} not supported for EmbeddingModelDocEnc")

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last non-padding token's hidden state."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode raw document tokens using the shared base_causallm + output_projection.

        Args:
            input_document_ids: (batch_size, num_documents, doc_seq_len)
                First half are positives, second half are negatives.
            attention_mask_document: (batch_size, num_documents, doc_seq_len)

        Returns:
            positive_embeddings: (batch_size, num_pos, embedding_dim)
            negative_embeddings: (batch_size, num_neg, embedding_dim)
        """
        batch_size, num_docs, doc_seq_len = input_document_ids.shape
        assert num_docs % 2 == 0, f"num_documents must be even (pos+neg), got {num_docs}"

        # Flatten to (batch_size * num_documents, doc_seq_len)
        flat_ids = input_document_ids.reshape(-1, doc_seq_len)
        flat_mask = attention_mask_document.reshape(-1, doc_seq_len)

        # Encode through the shared causal LM
        outputs = self.base_causallm(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        # Last-token pooling on the final layer hidden states
        doc_hidden = self.last_token_pool(outputs.last_hidden_state, flat_mask)
        # Project to retriever embedding space
        doc_embeddings = self.output_projection(doc_hidden)  # (batch_size * num_docs, embedding_dim)

        if self.normalize_embeddings:
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)

        # Reshape and split into positive / negative
        doc_embeddings = doc_embeddings.reshape(batch_size, num_docs, -1)
        num_pos = num_docs // 2
        positive_embeddings = doc_embeddings[:, :num_pos, :]
        negative_embeddings = doc_embeddings[:, num_pos:, :]
        return positive_embeddings, negative_embeddings

    def forward(self, **inputs):
        """
        Training forward pass with scheduled sampling.

        1. Encode documents via the shared LM (last-token pooling + output_projection)
        2. Run causal LM on query tokens for initial hidden states
        3. Autoregressive loop: generate query embeddings one at a time
        4. Contrastive loss between query embeddings and document embeddings
        """
        assert 'input_document_ids' in inputs, "Expected raw document token IDs"
        assert 'position_ids' in inputs, "position_ids is required"

        loss_mask = inputs['attention_mask'].detach().clone()

        # Step 1: Encode documents on-the-fly using the shared encoder
        positive_embeddings, negative_embeddings = self.encode_documents(
            inputs['input_document_ids'], inputs['attention_mask_document']
        )
        labels = positive_embeddings  # teacher forcing targets
        output_len = labels.size(1)

        # Step 2: Run causal LM on query tokens
        input_start_for_output = inputs['attention_mask'].sum(dim=1).max()
        assert inputs['attention_mask'][:, input_start_for_output:].sum().item() == 0

        initial_embeds = self.embedding(inputs['input_ids'])
        current_input = initial_embeds[:, :input_start_for_output, :]

        outputs = self.base_causallm(
            inputs_embeds=initial_embeds,
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
        )

        # Step 3: Autoregressive generation with scheduled sampling
        sampling_rate = inputs['sampling_rate']
        all_outputs = []
        for j in range(output_len):
            outputs = self.base_causallm(
                inputs_embeds=current_input,
                position_ids=inputs['position_ids'][:, :input_start_for_output + j],
                attention_mask=inputs['attention_mask'][:, :input_start_for_output + j],
            )

            next_emb = outputs.last_hidden_state[:, input_start_for_output + j - 1, :]
            all_outputs.append(next_emb.unsqueeze(1))

            # Scheduled sampling
            use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)
            predicted = self.input_projection(self.output_projection(next_emb)).unsqueeze(1)
            teacher = self.input_projection(labels[:, j].float()).unsqueeze(1)
            next_input = torch.where(use_predicted, predicted, teacher)

            current_input = torch.cat((current_input, next_input), dim=1)

            # Update position_ids and attention_mask for the new position
            inputs['position_ids'][:, input_start_for_output + j:] = inputs['position_ids'][:, input_start_for_output + j:] + 1
            inputs['attention_mask'][:, input_start_for_output + j] = 1

        # Step 4: Build loss mask
        loss_mask[:, :output_len] = 1
        loss_mask[:, output_len:] = 0
        assert loss_mask.float().mean(dim=0).sum().item() == output_len

        out_hidden_states = torch.cat(all_outputs, dim=1)

        # Step 5: Project hidden states → retriever embedding space
        if len(loss_mask.nonzero().size()) > 2:
            mask_indices = loss_mask.nonzero().squeeze()
        else:
            mask_indices = loss_mask.nonzero()

        selected_out_hidden_states = out_hidden_states[mask_indices[:, 0], mask_indices[:, 1]]
        selected_outputs_embeddings = self.output_projection(selected_out_hidden_states).contiguous()
        selected_outputs_embeddings = selected_outputs_embeddings.view(
            labels.size(0), labels.size(1), -1
        )
        assert selected_outputs_embeddings.size() == labels.size(), \
            (selected_outputs_embeddings.size(), labels.size())

        # Step 6: Contrastive loss
        loss = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
        return Outputs(
            loss=loss,
            inputs_embeds=current_input,
            last_hidden_states=selected_outputs_embeddings,
            labels=labels
        )

    def generate(self, max_new_tokens=16, **inputs):
        """
        Inference: autoregressively generate multiple query embeddings.
        """
        self.gen_forward_cnt = 0
        assert 'input_ids' in inputs or 'inputs_embeds' in inputs

        if 'input_ids' in inputs:
            assert inputs['input_ids'].shape[0] == 1, "only support batch_size == 1"
            assert inputs['input_ids'].size(1) == inputs['attention_mask'].sum()
            initial_embeds = self.embedding(inputs['input_ids'])
            outputs = self.base_causallm(
                inputs_embeds=initial_embeds,
                attention_mask=inputs['attention_mask'],
            )
        else:
            assert inputs['inputs_embeds'].shape[0] == 1, "only support batch_size == 1"
            assert inputs['inputs_embeds'].size(1) == inputs['attention_mask'].sum()
            initial_embeds = self.input_projection(inputs['inputs_embeds'].float())
            outputs = self.base_causallm(
                inputs_embeds=initial_embeds,
            )

        next_embs = []

        if max_new_tokens == 1:
            out_embs = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
            return self.output_projection(out_embs)

        next_emb = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
        next_embs.append(next_emb)
        new_inputs_embeds = torch.cat(
            (initial_embeds, self.input_projection(self.output_projection(next_emb))), dim=1
        )

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_emb = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
            next_embs.append(next_emb)
            new_inputs_embeds = torch.cat(
                (new_inputs_embeds, self.input_projection(self.output_projection(next_emb))), dim=1
            )

        out_embs = torch.cat(next_embs, dim=1)
        return self.output_projection(out_embs)

    def encode_documents_for_retrieval(self, input_ids, attention_mask):
        """Encode documents for building a retrieval index at inference time."""
        outputs = self.base_causallm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        doc_hidden = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        doc_embeddings = self.output_projection(doc_hidden)
        if self.normalize_embeddings:
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        return doc_embeddings


class EmbeddingModelDocEncNoProj(nn.Module):
    """
    Autoregressive multi-query embedding model with a shared encoder and NO
    input/output projection layers.

    Everything operates in the LM's native hidden_size space:
    - Documents: base_causallm + last-token pooling → hidden_size embeddings
    - Queries: autoregressive generation, hidden states fed back directly
    - Loss: contrastive loss computed in hidden_size space

    Expected batch keys (training):
        - input_ids: (batch_size, seq_len) — tokenized query
        - attention_mask: (batch_size, seq_len)
        - position_ids: (batch_size, seq_len)
        - input_document_ids: (batch_size, num_documents, doc_seq_len)
            First half are positive docs, second half are negative docs.
        - attention_mask_document: (batch_size, num_documents, doc_seq_len)
        - sampling_rate: float — scheduled sampling probability
    """

    def __init__(
        self,
        base_causallm,
        start_latent_id,
        eos_token_id,
        loss_function='Hungarian_Contrastive',
        temperature=0.05,
        normalize_embeddings=True
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.hidden_size = self.base_causallm.config.hidden_size
        self.normalize_embeddings = normalize_embeddings
        self.embedding = self.base_causallm.get_input_embeddings()

        if loss_function == 'Contrastive':
            self.loss_fct = ContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        elif loss_function == 'Hungarian_Contrastive':
            self.loss_fct = HungarianContrastiveLoss(temperature=temperature, normalize_embeddings=normalize_embeddings)
        else:
            raise ValueError(f"Loss function {loss_function} not supported for EmbeddingModelDocEncNoProj")

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Pool the last non-padding token's hidden state."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode raw document tokens using the shared base_causallm (no projection).

        Returns embeddings in hidden_size space.
        """
        batch_size, num_docs, doc_seq_len = input_document_ids.shape
        assert num_docs % 2 == 0, f"num_documents must be even (pos+neg), got {num_docs}"

        flat_ids = input_document_ids.reshape(-1, doc_seq_len)
        flat_mask = attention_mask_document.reshape(-1, doc_seq_len)

        outputs = self.base_causallm(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )
        # Last-token pooling → stays in hidden_size space, no projection
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        doc_embeddings = doc_embeddings.reshape(batch_size, num_docs, -1)
        num_pos = num_docs // 2
        positive_embeddings = doc_embeddings[:, :num_pos, :]
        negative_embeddings = doc_embeddings[:, num_pos:, :]
        return positive_embeddings, negative_embeddings

    def forward(self, **inputs):
        """
        Training forward pass with scheduled sampling, no projection layers.

        All embeddings (query and document) live in hidden_size space.
        Hidden states are fed back directly into the LM during the autoregressive loop.
        """
        assert 'input_document_ids' in inputs, "Expected raw document token IDs"
        assert 'position_ids' in inputs, "position_ids is required"

        loss_mask = inputs['attention_mask'].detach().clone()

        # Step 1: Encode documents (hidden_size space, no projection)
        positive_embeddings, negative_embeddings = self.encode_documents(
            inputs['input_document_ids'], inputs['attention_mask_document']
        )
        labels = positive_embeddings  # teacher forcing targets (hidden_size)
        output_len = labels.size(1)

        # Step 2: Run causal LM on query tokens
        input_start_for_output = inputs['attention_mask'].sum(dim=1).max()
        assert inputs['attention_mask'][:, input_start_for_output:].sum().item() == 0

        initial_embeds = self.embedding(inputs['input_ids'])
        current_input = initial_embeds[:, :input_start_for_output, :]

        outputs = self.base_causallm(
            inputs_embeds=initial_embeds,
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
        )

        # Step 3: Autoregressive generation with scheduled sampling
        # No projection — hidden states are fed back directly
        sampling_rate = inputs['sampling_rate']
        all_outputs = []
        for j in range(output_len):
            outputs = self.base_causallm(
                inputs_embeds=current_input,
                position_ids=inputs['position_ids'][:, :input_start_for_output + j],
                attention_mask=inputs['attention_mask'][:, :input_start_for_output + j],
            )

            next_emb = outputs.last_hidden_state[:, input_start_for_output + j - 1, :]
            all_outputs.append(next_emb.unsqueeze(1))

            # Scheduled sampling: predicted hidden state or teacher (doc hidden state)
            use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)
            predicted = next_emb.unsqueeze(1)                   # already in hidden_size
            teacher = labels[:, j].float().unsqueeze(1)         # also in hidden_size
            next_input = torch.where(use_predicted, predicted, teacher)

            current_input = torch.cat((current_input, next_input), dim=1)

            inputs['position_ids'][:, input_start_for_output + j:] = inputs['position_ids'][:, input_start_for_output + j:] + 1
            inputs['attention_mask'][:, input_start_for_output + j] = 1

        # Step 4: Build loss mask
        loss_mask[:, :output_len] = 1
        loss_mask[:, output_len:] = 0
        assert loss_mask.float().mean(dim=0).sum().item() == output_len

        out_hidden_states = torch.cat(all_outputs, dim=1)

        # Step 5: Extract query embeddings (no projection, stay in hidden_size)
        if len(loss_mask.nonzero().size()) > 2:
            mask_indices = loss_mask.nonzero().squeeze()
        else:
            mask_indices = loss_mask.nonzero()

        selected_out_hidden_states = out_hidden_states[mask_indices[:, 0], mask_indices[:, 1]]
        selected_outputs_embeddings = selected_out_hidden_states.contiguous()
        selected_outputs_embeddings = selected_outputs_embeddings.view(
            labels.size(0), labels.size(1), -1
        )
        assert selected_outputs_embeddings.size() == labels.size(), \
            (selected_outputs_embeddings.size(), labels.size())

        # Step 6: Contrastive loss (all in hidden_size space)
        loss = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
        return Outputs(
            loss=loss,
            inputs_embeds=current_input,
            last_hidden_states=selected_outputs_embeddings,
            labels=labels
        )

    def generate(self, max_new_tokens=16, **inputs):
        """
        Inference: autoregressively generate multiple query embeddings in hidden_size space.
        """
        self.gen_forward_cnt = 0
        assert 'input_ids' in inputs or 'inputs_embeds' in inputs

        if 'input_ids' in inputs:
            assert inputs['input_ids'].shape[0] == 1, "only support batch_size == 1"
            assert inputs['input_ids'].size(1) == inputs['attention_mask'].sum()
            initial_embeds = self.embedding(inputs['input_ids'])
            outputs = self.base_causallm(
                inputs_embeds=initial_embeds,
                attention_mask=inputs['attention_mask'],
            )
        else:
            assert inputs['inputs_embeds'].shape[0] == 1, "only support batch_size == 1"
            assert inputs['inputs_embeds'].size(1) == inputs['attention_mask'].sum()
            initial_embeds = inputs['inputs_embeds'].float()
            outputs = self.base_causallm(
                inputs_embeds=initial_embeds,
            )

        next_embs = []

        if max_new_tokens == 1:
            return outputs.last_hidden_state[:, -1, :].unsqueeze(1)

        next_emb = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
        next_embs.append(next_emb)
        new_inputs_embeds = torch.cat((initial_embeds, next_emb), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_emb = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
            next_embs.append(next_emb)
            new_inputs_embeds = torch.cat((new_inputs_embeds, next_emb), dim=1)

        return torch.cat(next_embs, dim=1)

    def encode_documents_for_retrieval(self, input_ids, attention_mask):
        """Encode documents for retrieval index (hidden_size space, no projection)."""
        outputs = self.base_causallm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        if self.normalize_embeddings:
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        return doc_embeddings


def load_model(
    train_lora, base_model_id, adapter_path, linear_checkpoint_path, embedding_model_dim,
    weight_tying=False, loss_function='Hungarian_Contrastive', temperature=0.05,
    lora_alpha=16, lora_r=64, lora_dropout=0.1,
    normalize_embeddings=True,
    model_type="EmbeddingModelDocEnc",
):
    # Load the base encoder model
    base_model = AutoModel.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # Load the tokenizer
    if 'full_finetuning' in base_model_id:
        if 'inf' in base_model_id.split('/')[1]:
            tokenizer = AutoTokenizer.from_pretrained("infly/inf-retriever-v1-1.5b")
        elif 'llama' in base_model_id.split('/')[1]:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        else:
            raise ValueError(f"Invalid base model id: {base_model_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    start_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    # Apply LoRA or full fine-tuning
    if train_lora:
        if adapter_path is not None:
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
    else: # full finetuning
        if adapter_path is not None: 
            # Load the model with LoRA adapter weights, but set all parameters to trainable
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

    # Wrap into the requested model class
    if model_type == "EmbeddingModelDocEnc":
        print(f"loading model {model_type}")
        model = EmbeddingModelDocEnc(
            base_causallm=model,
            start_latent_id=start_id,
            eos_token_id=tokenizer.eos_token_id,
            embedding_model_dim=embedding_model_dim,
            weight_tying=weight_tying,
            loss_function=loss_function,
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
        )
        # Load linear projection layers if provided
        if linear_checkpoint_path is not None:
            print(f"loading linear layers from {linear_checkpoint_path}")
            linear_layers = torch.load(linear_checkpoint_path)
            model.input_projection.load_state_dict(linear_layers['input_projection'])
            model.output_projection.load_state_dict(linear_layers['output_projection'])
    elif model_type == "EmbeddingModelDocEncNoProj":
        print(f"loading model {model_type}")
        model = EmbeddingModelDocEncNoProj(
            base_causallm=model,
            start_latent_id=start_id,
            eos_token_id=tokenizer.eos_token_id,
            loss_function=loss_function,
            temperature=temperature,
            normalize_embeddings=normalize_embeddings,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported in model_train_enc")

    return model, tokenizer


def save_model_distributed(model, save_dir, step, eval_loss, accelerator, logger, save_best_model=False):
    state_dict = accelerator.get_state_dict(model)

    has_projections = 'input_projection.weight' in state_dict
    if accelerator.is_main_process and has_projections:
        linear_layers = {
            'input_projection': OrderedDict({'weight': state_dict['input_projection.weight']}),
            'output_projection': OrderedDict({'weight': state_dict['output_projection.weight']}),
            'step': step,
            'loss': eval_loss
        }
    else:
        linear_layers = None

    if save_best_model:
        base_save_dir = os.path.join(save_dir, "best_model")
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

    if linear_layers is not None:
        suffix = "best_model_linear.pt" if save_best_model else f"checkpoint_{step}_linear.pt"
        accelerator.save(linear_layers, os.path.join(save_dir, suffix))
        logger.info(f"saving model.", step=(step), process_index=accelerator.process_index)


def save_model_single(model, save_dir, step, eval_loss, logger, save_best_model=False):
    if save_best_model:
        base_save_dir = os.path.join(save_dir, "best_model")
    else:
        base_save_dir = os.path.join(save_dir, f"checkpoint_{step}")

    model.base_causallm.save_pretrained(base_save_dir, safe_serialization=True)

    if hasattr(model, 'input_projection'):
        linear_layers = {
            'input_projection': model.input_projection.state_dict(),
            'output_projection': model.output_projection.state_dict(),
            'step': step,
            'loss': eval_loss
        }
        suffix = "best_model_linear.pt" if save_best_model else f"checkpoint_{step}_linear.pt"
        torch.save(linear_layers, os.path.join(save_dir, suffix))

    logger.info(f"saving model.", step=(step))
