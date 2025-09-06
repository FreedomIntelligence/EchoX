from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CTCLoss
import transformers

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def padding_tensor(tensor, length, dim=0, pad=False):

    if length == 0:
        return tensor
    
    assert length > 0, f"Wrong padding length: {length}"

    shape = list(tensor.shape)
    assert dim < len(shape), f"dim {dim} out of shape {shape}"
    shape[dim] = length
    padding_tensor = torch.cat(
        (
            tensor,
            torch.full(tuple(shape), pad, dtype=tensor.dtype, device=tensor.device)
        ), 
        dim=dim
    )
    return padding_tensor


class T2ULlamaConfig(LlamaConfig):
    model_type = "T2ULlama"
    
class T2ULlamaForCausalLM(LlamaForCausalLM):
    config_class = T2ULlamaConfig

    def __init__(self, config, embedding_weight=None):
        
        self.current_step = 0
        self.log = {}

        super(LlamaForCausalLM, self).__init__(config)
        self.config = config
        self.training_stage = config.unit_output
        self.pad_token_id = 128009

        llama_config = T2ULlamaConfig(**config.to_dict(), 
                                        batch_first=True, 
                                        norm_first=True
                                    )
        llama_config.architectures = ["T2ULlamaForCausalLM"]
        llama_config.pad_token_id = self.pad_token_id
        llama_config.vocab_size += llama_config.unit_vocab_size
        #######################################################
        llama_config.unit_model = "medium"
        llama_config.max_position_embeddings = 2048     # 1024 1536 2048       # origin 1024 reduced 512
        #######################################################
        if hasattr(llama_config, "unit_model"):
            if llama_config.unit_model == "large":
                llama_config.num_hidden_layers = 2
                # llama_config.hidden_size = 4096
                # llama_config.num_attention_heads = 32
                # llama_config.intermediate_size = 14336
                # llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads

            elif llama_config.unit_model == "tiny":
                llama_config.num_hidden_layers = 4
                llama_config.hidden_size = 512
                llama_config.num_attention_heads = 8
                llama_config.intermediate_size = 2048
                llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
            else:
                llama_config.num_hidden_layers = 8
                llama_config.hidden_size = 768
                llama_config.num_attention_heads = 12
                llama_config.num_key_value_heads = 12
                llama_config.intermediate_size = 2048
                llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
        else:
            llama_config.num_hidden_layers = 6
            llama_config.hidden_size = 512
            llama_config.num_attention_heads = 8
            llama_config.intermediate_size = 2048
            llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
        # print(llama_config)
        
        self.model = LlamaModel(llama_config)
        # share embedding 0501 by kkq
        self.model.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.pad_token_id)   # redefine
        self.unit_embedding = nn.Linear(config.hidden_size, llama_config.unit_vocab_size, bias=False) 
        self.adapter = nn.Linear(config.hidden_size, llama_config.hidden_size, bias = True) 
        self.lm_head = nn.Linear(llama_config.hidden_size, llama_config.vocab_size, bias=False)

        if self.training_stage == "pretrain":
            pass
        elif self.training_stage == "finetune" or self.training_stage == "finetune_kd" or self.training_stage == "finetune_kd_online":
            self.aligner_MLP = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )
            torch.nn.init.ones_(self.aligner_MLP[0].weight)
            torch.nn.init.zeros_(self.aligner_MLP[0].bias)
            torch.nn.init.ones_(self.aligner_MLP[3].weight)
            torch.nn.init.zeros_(self.aligner_MLP[3].bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def insert_text_embedding(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        shift_text_labels: Optional[torch.LongTensor] = None,
        shift_text_hidden_states: Optional[torch.FloatTensor] = None,
        unit_targets: Optional[torch.LongTensor] = None,
        sub_lengths: Optional[torch.LongTensor] = None,
        text_start_index: Optional[torch.LongTensor] = None,
        do_task: str = None,
        **kwargs: dict,
    ):  

        if inputs_embeds == None:
            # share embedding 0501 by kkq
            embed_tokens_weight = torch.cat(
                [
                    self.model.embed_tokens.weight.detach(), self.unit_embedding.weight
                ],
                dim = 0,
            )
            # print(embed_tokens_weight, embed_tokens_weight.shape)
            inputs_embeds = F.embedding(input_ids, embed_tokens_weight, padding_idx=self.pad_token_id)
        
        emb_loss = None
        if do_task == "pretrain":
            if self.training:
                if hasattr(self, "embedding_dropout"):
                    emb_origin_mask = text_labels != -100
                    origin_padding_length = labels.shape[-1] - emb_origin_mask.shape[-1]
                    extend_emb_origin_mask = padding_tensor(emb_origin_mask, origin_padding_length, 1, False)
                    extend_emb_origin_mask = ~extend_emb_origin_mask.unsqueeze(-1).expand_as(inputs_embeds)

                    # Π-Model + noise
                    log_var = self.perturb(inputs_embeds)
                    perturbed_inputs_embeds_2 = inputs_embeds + torch.randn_like(inputs_embeds) * (torch.exp(0.5 * log_var) + 1e-6)
                    # Π-Model + dropout
                    perturbed_inputs_embeds_1 = self.embedding_dropout(inputs_embeds)
                    perturbed_inputs_embeds_2 = self.embedding_dropout(perturbed_inputs_embeds_2)
                    perturbed_inputs_embeds_1 = torch.where(extend_emb_origin_mask, inputs_embeds, perturbed_inputs_embeds_1)
                    perturbed_inputs_embeds_2 = torch.where(extend_emb_origin_mask, inputs_embeds, perturbed_inputs_embeds_2)

                    inputs_embeds = torch.cat(
                        (perturbed_inputs_embeds_1, perturbed_inputs_embeds_2),
                        dim=0,
                    )

                    kl_loss = -0.5 * (1 + log_var - log_var.exp()).mean(dim=-1).sum(dim=-1).mean()
                    contrastive_loss = (1 - F.cosine_similarity(perturbed_inputs_embeds_1, perturbed_inputs_embeds_2, dim=-1)).sum(dim=-1).mean()
                    emb_loss = kl_loss + contrastive_loss

                    if kl_loss.device == torch.device("cuda:0"):
                        self.log["kl_loss"] = kl_loss.item()
                        self.log["std"] = torch.exp(0.5 * log_var).mean().item()
                        self.log["contrastive_loss"] = contrastive_loss.item()

            pass
        elif do_task == "finetune":
            inputs_embeds = inputs_embeds.detach()
            inputs_embeds_refer = inputs_embeds.clone().detach()
            shift_text_hidden_states = self.aligner_MLP(shift_text_hidden_states)
            emb_origin_mask = text_labels != -100    # get output text pos
            emb_shift_mask = shift_text_labels != -100

            origin_padding_length = labels.shape[-1] - emb_origin_mask.shape[-1]
            shift_padding_length = labels.shape[-1] - emb_shift_mask.shape[-1]
            
            extend_emb_origin_mask = padding_tensor(emb_origin_mask, origin_padding_length, 1, False)
            extend_emb_shift_mask = padding_tensor(emb_shift_mask, shift_padding_length, 1, False)
            extend_shift_text_hidden_states = padding_tensor(shift_text_hidden_states, shift_padding_length, 1, 1e-9)
            # check
            extend_text_labels = padding_tensor(text_labels, origin_padding_length, 1, -100)
            extend_shift_text_labels = padding_tensor(shift_text_labels, shift_padding_length, 1, -100)
            
            assert torch.equal(
                extend_text_labels[extend_emb_origin_mask], 
                extend_shift_text_labels[extend_emb_shift_mask]
            ), "{}\n{}\n{}\n{}".format(labels, extend_emb_origin_mask, extend_shift_text_labels, extend_emb_shift_mask)
            
            inputs_embeds[extend_emb_origin_mask.unsqueeze(-1).expand_as(inputs_embeds)] = \
                extend_shift_text_hidden_states[extend_emb_shift_mask.unsqueeze(-1).expand_as(extend_shift_text_hidden_states)].to(dtype=inputs_embeds.dtype)
            
            if self.training:
                contrastive_loss = (1 - F.cosine_similarity(inputs_embeds, inputs_embeds_refer, dim=-1)).sum(-1).mean()
                emb_loss = contrastive_loss
                if emb_loss.device == torch.device("cuda:0"):
                    self.log["contrastive_loss"] = contrastive_loss.item()
                pass
        elif do_task == "finetune_kd" :
            #inputs_embeds = inputs_embeds.detach()
            #inputs_embeds_refer = inputs_embeds.clone().detach()
            #print(text_labels)
            #print(sub_lengths.sum())
            emb_origin_mask = text_labels != -100

            fetch_lables_list = [] 
            for batch in range(emb_origin_mask.shape[0]):
                fetch_lables_list.append(text_labels[batch][emb_origin_mask[batch]])
            shift_text_hidden_states = self.aligner_MLP(shift_text_hidden_states)

            #split the shift_text_hidden_states
            #[128006, 128000, 78191, 128007, 128000, 198, 128000]
            maxn_length = sub_lengths.max() + 8
            pad_ids = torch.full(size=(sub_lengths.shape[0], sub_lengths.shape[1], maxn_length), fill_value=self.pad_token_id, dtype=torch.long).to(shift_text_hidden_states.device)

            pad_text_ids = torch.full(size=(sub_lengths.shape[0], sub_lengths.shape[1], maxn_length), fill_value=self.pad_token_id, dtype=torch.long).to(shift_text_hidden_states.device)

            atten_mask = pad_ids.ne(self.pad_token_id)
            #target_mask_part1 = pad_ids.ne(self.pad_token_id)
            shift_text_hidden_states_slice = F.embedding(pad_ids, embed_tokens_weight, padding_idx=self.pad_token_id)
            
            #print(shift_text_hidden_states_slice.shape,shift_text_hidden_states.shape)
            for batch in range(sub_lengths.shape[0]):
                cot=0
                start_index = text_start_index[batch]
                for index, sub_length in enumerate(sub_lengths[batch]):
                    if sub_length==-1:
                        break
                    #print(shift_text_hidden_states_slice[batch][index][:sub_length].shape, shift_text_hidden_states[batch][cot:cot+sub_length].shape)
                    eos_id = torch.IntTensor([128009]).to(inputs_embeds.device)
                    eos = self.model.embed_tokens(eos_id)
                    if index == 0:
                        text_prefix_ids = torch.IntTensor([128006, 128000, 65576, 128007, 128000, 198]).to(inputs_embeds.device)
                        preifx_embed = self.model.embed_tokens(text_prefix_ids)
                        pad_text_ids[batch][index][:sub_length+7] = torch.cat([text_prefix_ids, fetch_lables_list[batch][cot:cot+sub_length], eos_id],dim=0)
                        atten_mask[batch][index][:sub_length+7]=True
                    else:
                        text_prefix_ids = torch.IntTensor([128006, 128000, 65576, 128007, 128000, 198, 12800]).to(inputs_embeds.device)
                        preifx_embed = self.model.embed_tokens(text_prefix_ids)
                        pad_text_ids[batch][index][:sub_length+8] = torch.cat([text_prefix_ids, fetch_lables_list[batch][cot:cot+sub_length], eos_id], dim=0)
                        atten_mask[batch][index][:sub_length+8]=True

                    new_shift_text_hidden_states = torch.cat([preifx_embed, shift_text_hidden_states[batch][start_index+cot:start_index+cot+sub_length], eos], dim = 0)
                    shift_text_hidden_states_slice[batch][index][:new_shift_text_hidden_states.shape[0]] = new_shift_text_hidden_states

                    cot+=sub_length
            shift_text_hidden_states_slice = shift_text_hidden_states_slice.reshape(shift_text_hidden_states_slice.shape[0]*shift_text_hidden_states_slice.shape[1],shift_text_hidden_states_slice.shape[2],shift_text_hidden_states_slice.shape[3])    


            padding_unit_targets = unit_targets.clone()
            padding_unit_targets = torch.where(padding_unit_targets == IGNORE_TOKEN_ID, self.pad_token_id, padding_unit_targets)
            target_mask_part = padding_unit_targets.ne(self.pad_token_id)
            atten_mask = torch.cat([atten_mask, target_mask_part], dim = -1)
            atten_mask = atten_mask.reshape(atten_mask.shape[0]*atten_mask.shape[1],atten_mask.shape[2])

            pad_text_ids = pad_text_ids.reshape(pad_text_ids.shape[0]*pad_text_ids.shape[1],pad_text_ids.shape[2])
            shift_text_embeddings = F.embedding(pad_text_ids, embed_tokens_weight, padding_idx=self.pad_token_id)

            unit_target_slice = F.embedding(padding_unit_targets, embed_tokens_weight, padding_idx=self.pad_token_id)
            # unit_target_slice = F.embedding(unit_targets, embed_tokens_weight, padding_idx=self.pad_token_id)             
            unit_target_slice = unit_target_slice.reshape(unit_target_slice.shape[0]*unit_target_slice.shape[1],unit_target_slice.shape[2],unit_target_slice.shape[3])

            inputs_embeds = torch.cat([shift_text_hidden_states_slice, unit_target_slice], dim = 1)

            ignore_ids = torch.full(size=(sub_lengths.shape[0], sub_lengths.shape[1], maxn_length), fill_value=IGNORE_TOKEN_ID, dtype=torch.long).to(shift_text_hidden_states.device)
            unit_targets = torch.cat([ignore_ids,unit_targets],dim=-1) 
            unit_targets = unit_targets.reshape(unit_targets.shape[0]*unit_targets.shape[1],unit_targets.shape[2])

            if self.training:
                #print(shift_text_hidden_states_slice.shape, shift_text_embeddings.shape)
                contrastive_loss = (1 - F.cosine_similarity(shift_text_hidden_states_slice, shift_text_embeddings, dim=-1)).sum(-1).mean()
                emb_loss = contrastive_loss
                if emb_loss.device == torch.device("cuda:0"):
                    self.log["contrastive_loss"] = contrastive_loss.item()

        elif do_task == "finetune_kd_online":
            shift_text_hidden_states = self.aligner_MLP(shift_text_hidden_states)
            gold_inputs_embeds = inputs_embeds.clone()
            for batch in range (inputs_embeds.shape[0]):
                start_index = text_start_index[batch]
                for slice_index in range (inputs_embeds.shape[1]):
                    sub_length= sub_lengths[batch][slice_index]
                    inputs_embeds[batch][slice_index][7:7+sub_length] = shift_text_hidden_states[batch][start_index+1:start_index+1+sub_length]
                    start_index += sub_length
            if self.training:
                #print(shift_text_hidden_states_slice.shape, shift_text_embeddings.shape)
                contrastive_loss = ((1 - F.cosine_similarity(inputs_embeds, gold_inputs_embeds, dim=-1)) * attention_mask).sum(-1).mean()
                emb_loss = contrastive_loss
                if emb_loss.device == torch.device("cuda:0"):
                    self.log["contrastive_loss"] = contrastive_loss.item()
            unit_embeds = F.embedding(unit_targets, embed_tokens_weight, padding_idx=self.pad_token_id)

            inputs_embeds = torch.cat([inputs_embeds,unit_embeds], dim=2)
        else:
            inputs_embeds = self.aligner_MLP(inputs_embeds)
            #[start_header_id] + _speaker + [end_header_id] + nl_tokens only for batch one!
            units_ids = torch.IntTensor([[128009, 128006, 128000, 65576, 128007, 128000, 198]]).to(inputs_embeds.device)
            units_prefix = self.model.embed_tokens(units_ids)
            text_ids = torch.IntTensor([[128006, 128000, 65576, 128007, 128000, 198, 12800]]).to(inputs_embeds.device)
            text_prefix = self.model.embed_tokens(text_ids)
            inputs_embeds = torch.cat([text_prefix, inputs_embeds, units_prefix], dim = 1)

        inputs_embeds = self.adapter(inputs_embeds)
        if do_task == "finetune_kd":
            return (emb_loss, inputs_embeds, unit_targets, atten_mask,)
        else:
            return (emb_loss, inputs_embeds)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds == None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            # share embedding 0501 by kkq
            embed_tokens_weight = torch.cat(
                [
                    self.model.embed_tokens.weight.detach(), self.unit_embedding.weight
                ],
                dim = 0,
            )
            # print(embed_tokens_weight, embed_tokens_weight.shape)
            inputs_embeds = F.embedding(input_ids, embed_tokens_weight, padding_idx=self.pad_token_id)
            inputs_embeds = self.adapter(inputs_embeds)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        cr_loss = None
        if labels != None:
            shift_labels = labels

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = shift_labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
        
            shift_logits = shift_logits.view(-1, (self.config.vocab_size + self.config.unit_vocab_size))
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
                
            if loss.device == torch.device("cuda:0"):
                self.log["unit_loss"] = loss.item()

            if cr_loss != None:
                target_scale = loss.item() * 0.2
                cr_loss_weight = target_scale / cr_loss.item() if cr_loss > target_scale else 1.0
                loss = loss + cr_loss_weight * cr_loss

            if loss.device == torch.device("cuda:0") and (self.current_step - 10) % 100 == 0:
                print(self.log, loss.device)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoConfig.register("T2ULlama", T2ULlamaConfig)
AutoModelForCausalLM.register(T2ULlamaConfig, T2ULlamaForCausalLM)