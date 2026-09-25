# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import transformers
from transformers import BertModel, XLMRobertaModel

from src import utils, inbatch


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb



def load_retriever(model_path, pooling="average", random_init=False):
    # try: check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # retriever_model_id = "bert-base-uncased"
            retriever_model_id = "bert-base-multilingual-cased"
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)

        if hasattr(opt, "run_name"):
            if hasattr(opt, "training_mode"):
                print(opt.training_mode)
                if opt.training_mode in ['standard']:
                    model_class = inbatch.InBatch
                elif opt.training_mode == 'subtraction':
                    model_class = inbatch.InBatchSubtraction
                elif opt.training_mode == 'gru':
                    model_class = inbatch.InBatchGRU
                elif opt.training_mode == 'concat':
                    model_class = inbatch.InBatchConcat
                elif opt.training_mode == 'subtraction_linear':
                    model_class = inbatch.InBatchSubtractionLinear
                elif opt.training_mode == 'linear_projection':
                    model_class = inbatch.InBatchLinearProjection
                elif opt.training_mode == 'sentence_transformer':
                    model_class = inbatch.InBatchSentenceTransformer
                else:
                    model_class = inbatch.InBatch
            else:
                raise NotImplementedError
        else:
            model_class = Contriever
            
        print(model_class, retriever_model_id)
        pretrained_dict = pretrained_dict["model"]
        if model_class in [Contriever]:
            retriever = model_class(cfg)
            retriever.load_state_dict(pretrained_dict, strict=True)
        else:
            # NOTE: added this
            cfg = utils.load_hf(transformers.AutoConfig, opt.retriever_model_id)
            tokenizer = utils.load_hf(transformers.AutoTokenizer, opt.retriever_model_id)
            
            # Create a temporary Contriever instance
            temp_retriever = Contriever(cfg)  # Contriever is defined in this file

            if opt.training_mode in ['standard']:
                #model = inbatch.InBatch(opt, None, None)
                model = inbatch.InBatch(opt, temp_retriever, tokenizer)
                print("Using model = InBatch", flush=True)
            elif opt.training_mode == 'gru':
                model = inbatch.InBatchGRU(opt, None, None)
                print("Using model = InBatchGRU", flush=True)
            elif opt.training_mode == 'linear_projection':
                model = inbatch.InBatchLinearProjection(opt, None, None)
                print("Using model = InBatchLinearProjection", flush=True)
            else:
                model = inbatch.InBatch(opt, None, None)
            
            print(f'Before checkpoint - weight sum: {model.encoder.embeddings.word_embeddings.weight.sum().item()}',flush=True)

            model.load_state_dict(pretrained_dict, strict=True)

            # After loading checkpoint  
            print(f'✓ Loaded checkpoint with {len(pretrained_dict)} parameters',flush=True)
            print(f'✓ After checkpoint - weight sum: {model.encoder.embeddings.word_embeddings.weight.sum().item()}',flush=True)
            model.eval()
            print('finish getting model')
            print(f'✓ Successfully loaded finetuned checkpoint from {model_path}', flush=True)
            print(f'✓ Checkpoint weights loaded: {len(pretrained_dict)} parameters', flush=True)
            if opt.training_mode in ['standard']:
                # get the encoder
                retriever = model.encoder
            else:
                retriever = model
        
    else:
        # Not Loading From Local Checkpoint
        # Loading from Huggingface
        retriever_model_id = model_path
        model_class = Contriever
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(model_class, model_path)

    return retriever, tokenizer, retriever_model_id

# def load_retriever(model_path, pooling="average", random_init=False):
#     # try: check if model exists locally
#     path = os.path.join(model_path, "checkpoint.pth")
#     if os.path.exists(path):
#         pretrained_dict = torch.load(path, map_location="cpu")
#         opt = pretrained_dict["opt"]
#         if hasattr(opt, "retriever_model_id"):
#             retriever_model_id = opt.retriever_model_id
#         else:
#             # retriever_model_id = "bert-base-uncased"
#             retriever_model_id = "bert-base-multilingual-cased"
#         tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
#         cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
#         if hasattr(opt, "run_name"):
#             model_class = inbatch.InBatch
#         else:
#             model_class = Contriever
            
#         print(model_class, retriever_model_id)
#         pretrained_dict = pretrained_dict["model"]
#         if model_class in [Contriever]:
#             retriever = model_class(cfg)
#             retriever.load_state_dict(pretrained_dict, strict=True)
#         else:
#             model = inbatch.InBatch(opt, None, None)
#             model.load_state_dict(pretrained_dict, strict=True)
#             model.eval()
#             print('finish getting model')
#             # get the encoder
#             retriever = model.encoder
#     else:
#         # if not, load from HuggingFace
#         retriever_model_id = model_path
#         model_class = Contriever
        
#         cfg = utils.load_hf(transformers.AutoConfig, model_path)
#         tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
#         retriever = utils.load_hf(model_class, model_path)

#     return retriever, tokenizer, retriever_model_id
