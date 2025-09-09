from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CTCLoss


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers import (
    WhisperProcessor,
    WhisperModel,
)
from .t2ullama_cr_online import T2ULlamaForCausalLM

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class EchoXConfig(LlamaConfig):
    model_type = "EchoX"
    
def load_whisper(audio_tower_name, device="cuda"):
    model = WhisperModel.from_pretrained(
            audio_tower_name,torch_dtype=torch.float16,low_cpu_mem_usage=True).to(device)
    model.config.forced_decoder_ids = None
    return model

class LookBackModule(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.encoder_attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        self.atten_layer_norm = nn.LayerNorm(cfg.hidden_size)


    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
            #attn_mask=padding_mask,
        )
        x += residual
        x = self.atten_layer_norm(x)
        return x

class EchoXModel(LlamaModel):
    config_class = EchoXConfig

    def __init__(self, config: LlamaConfig):
        super(EchoXModel, self).__init__(config)

        if hasattr(config, "audio_tower"):
            self.audio_tower = [load_whisper(config.audio_tower)]

        if hasattr(config, "adapter_size"):

            self.mm_projector1 = nn.Linear(config.adapter_size*2 , config.hidden_size)
            asr_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size*2,
                dropout=0.1,
                norm_first=True
            )
            self.lbm =  LookBackModule(config)
            self.out_norm = nn.LayerNorm(config.hidden_size)
            self.audio_feature_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.asr_transformer_encoder = nn.TransformerEncoder(asr_encoder_layer, num_layers=1)
            self.mask_tensor=(torch.ones([1, 2048])>0)
            self.length=-1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaAA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        audio_tower = getattr(self, 'audio_tower', None)
        if audio_tower is not None and (input_ids.shape[1] != 1 or self.training) and audios is not None:
            audio_tower = audio_tower[0]  # HACK: for FSDP
            audio_list=[]
            
            audio_config = audio_tower.config
            for audio in audios:
                with torch.no_grad():
                    audio_feature = audio_tower.encoder(audio).last_hidden_state
           
                audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1]//2, 2 * audio_feature.shape[2])
                audio_feature = self.mm_projector1(audio_feature)
                audio_feature = self.asr_transformer_encoder(audio_feature)
                audio_feature = self.out_norm(audio_feature)
                audio_list.append(audio_feature)

            audio_features = torch.stack(audio_list, dim=0)
            batch = audio_features.shape[0]
            audio_turn = audio_features.shape[1]
            audio_features = audio_features.view((batch * audio_turn,)+audio_features.shape[2:])
 
            predict_logits = self.audio_feature_head(audio_features)

            new_input_embeds = []
            label_shift = []
            speech_pos = []
            label_extend = -1
            new_input_ids = []
            tokens = predict_logits.argmax(dim=-1)
            shrink_mask = tokens.roll(1) != tokens
            shrink_mask[:,0] = True
                
            lengths = shrink_mask.long().sum(-1)
            shrink_2d = audio_features[shrink_mask]
            #num_patches = audio_features.shape[1]
            num_patches = audio_config.audio_patch_size
            l_index=0
            shrink_features_raw = []
            for v, audio_feature, mask in zip(lengths, audio_features, ~shrink_mask):
                shrink_feature = shrink_2d[l_index:l_index+v]
                shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=mask)
                shrink_features_raw.append(shrink_feature)
                l_index += v

            shrink_features = []
            for i in range(0, len(shrink_features_raw), audio_turn):
                shrink_features.append(shrink_features_raw[i:i+audio_turn])  
            if self.training: 
                maxn_length = lengths.view(batch,audio_turn).sum(-1).max()
                label_extend = maxn_length - num_patches * audio_turn
                old_seq_length = inputs_embeds.shape[1]
                for cur_input_ids, cur_input_embeds, cur_shrink_features in zip(input_ids, inputs_embeds, shrink_features):
                    pad_ids = torch.full(size=(maxn_length,), fill_value=audio_config.llm_pad_token_id, dtype=torch.long).to(attention_mask.device)
                    pad_embeds = self.embed_tokens(pad_ids)
                    audio_start_token_pos_all = torch.where(cur_input_ids == audio_config.audio_patch_token)[0]
                    #print(cur_input_embeds.shape,cur_input_ids.shape)
                    inner_label_shift = []
                    inner_speech_pos = []
                    for audio_start_token_pos, shrink_feature in reversed(list(zip(audio_start_token_pos_all, cur_shrink_features))): #zip(audio_start_token_pos_all, cur_shrink_features):
                        cur_speech_length = shrink_feature.shape[0]
 
                        cur_input_ids = torch.cat((cur_input_ids[:audio_start_token_pos], 
                          cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(cur_speech_length), 
                          cur_input_ids[audio_start_token_pos + num_patches:]), dim=0)
                        cur_input_embeds = torch.cat((
                          cur_input_embeds[:audio_start_token_pos],
                          shrink_feature,
                          cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)
                        inner_label_shift.insert(0, cur_speech_length - num_patches)
                        inner_speech_pos.insert(0, audio_start_token_pos)

                    label_shift = label_shift + inner_label_shift
                    speech_pos = speech_pos + inner_speech_pos

                    cur_new_input_embeds = torch.cat((cur_input_embeds, pad_embeds[:old_seq_length + label_extend - cur_input_embeds.shape[0]]),dim=0)
                    cur_new_input_ids = torch.cat((cur_input_ids, pad_ids[:old_seq_length + label_extend - cur_input_ids.shape[0]]),dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    new_input_ids.append(cur_new_input_ids)
                    
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask=input_ids.ne(audio_config.llm_pad_token_id)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)

                batch_label_shift = []
                batch_speech_pos=[]
                for i in range(0, len(label_shift), audio_turn):
                    batch_label_shift.append(label_shift[i:i+audio_turn])
                    batch_speech_pos.append(speech_pos[i:i+audio_turn])
            else:
                # Inference mode with batch_size=1
                assert input_ids.shape[0] == 1, "This implementation only supports batch_size=1 during inference"
                
                # Get all audio token positions in this sample
                audio_start_token_positions = torch.where(input_ids[0] == audio_config.audio_patch_token)[0]
                
                # Initialize with original embeddings
                current_embeds = inputs_embeds[0]  # [seq_len, embed_dim]
                current_ids = input_ids[0]         # [seq_len]
                
                # Process each audio token position sequentially
                position_shift = 0  # Track position changes due to expansions
                
                # Ensure shrink_features is properly formatted
                if isinstance(shrink_features[0], list):
                    # If it's a list of lists (batch_size=1 but multiple turns), flatten it
                    shrink_features = [item for sublist in shrink_features for item in sublist]
                
                for pos_idx, audio_pos in enumerate(audio_start_token_positions):
                    adjusted_pos = audio_pos + position_shift
                    
                    # Get corresponding shrink feature (ensure it's a tensor)
                    shrink_feature = shrink_features[pos_idx]
                    if isinstance(shrink_feature, list):
                        shrink_feature = torch.stack(shrink_feature, dim=0)
                    
                    v = shrink_feature.shape[0]  # Now this should work
                    # print('len: ', v)
                    
                    # Expand the input ids and embeddings
                    current_ids = torch.cat([
                        current_ids[:adjusted_pos],
                        current_ids[adjusted_pos:adjusted_pos+1].repeat(v),
                        current_ids[adjusted_pos + num_patches:]
                    ], dim=0)
                    
                    current_embeds = torch.cat([
                        current_embeds[:adjusted_pos],
                        shrink_feature,
                        current_embeds[adjusted_pos + num_patches:]
                    ], dim=0)
                    
                    # Update position shift for next iteration
                    position_shift += (v - num_patches)
                
                # Update the tensors (unsqueeze to restore batch dim)
                input_ids = current_ids.unsqueeze(0)          # [1, new_seq_len]
                inputs_embeds = current_embeds.unsqueeze(0)   # [1, new_seq_len, embed_dim]
                attention_mask = input_ids.ne(audio_config.llm_pad_token_id)
                
                # Update inference state tracking
                if not hasattr(self, 'mask_tensor'):
                    # Initialize with current attention mask
                    self.mask_tensor = attention_mask.clone()
                    self.length = attention_mask.shape[1]
                else:
                    # Ensure mask tensor is on correct device
                    self.mask_tensor = self.mask_tensor.to(attention_mask.device)
                    
                    # Expand mask tensor if needed
                    if self.mask_tensor.shape[1] < attention_mask.shape[1]:
                        new_mask = torch.zeros(1, attention_mask.shape[1], 
                                            dtype=torch.bool,
                                            device=attention_mask.device)
                        new_mask[0, :self.mask_tensor.shape[1]] = self.mask_tensor
                        self.mask_tensor = new_mask
                    
                    # Update mask tensor
                    self.mask_tensor[0, :attention_mask.shape[1]] = attention_mask[0]
                    self.length = attention_mask.shape[1]
        
        attention_mask=self.mask_tensor[:,:self.length]
        self.length+=1
        
        return_state=super(EchoXModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.training and audios is not None:
            return_state["audio_features"] =  predict_logits
            return_state["label_shift"] = batch_label_shift
            return_state["label_extend"] = label_extend
            return_state["speech_pos"] = batch_speech_pos
        #return_state = {"audio_features":predict_logits}
        return return_state 


class EchoXForCausalLM(LlamaForCausalLM):
    config_class = EchoXConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = EchoXModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # t2u by kkq
        if hasattr(config, "unit_output"):
            self.unit_output = config.unit_output
            self.unit_translator = T2ULlamaForCausalLM(config, self.lm_head.weight)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_unit_translator(self):
        return self.unit_translator

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t2u_input_ids: Optional[torch.LongTensor] = None,
        t2u_labels: Optional[torch.LongTensor] = None,
        t2u_attention_mask: Optional[torch.Tensor] = None,
        unit_targets: Optional[torch.Tensor] = None,
        sub_lengths: Optional[torch.Tensor] = None,
        asr_targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        do_task: str = None,
        assistant_after_audio_shifts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        # t2u by kkq
        # pretrain(t2u only) finetune(s2t&e2u)
        do_task = do_task if do_task != None  else getattr(self, 'unit_output', None)

        outputs = None
        hidden_states = None
        new_shift_labels = None
        if do_task != "pretrain":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                audios=audios
            )
        

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and do_task != "pretrain" and do_task != "finetune_kd":
            if asr_targets is not None:
                asr_logits = outputs["audio_features"]
                asr_targets = asr_targets.view(asr_targets.shape[0] * asr_targets.shape[1], asr_targets.shape[2])
                mask_asr_targets = (asr_targets != IGNORE_TOKEN_ID)
                target_lengths = mask_asr_targets.sum(1)
                input_lengths = torch.full(size=(asr_logits.shape[0],), fill_value=asr_logits.shape[1], dtype=torch.long)

                loss_ctc = CTCLoss()

                log_probs = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)
                #print(asr_targets.shape)
                #print(input_lengths, target_lengths)

                with torch.backends.cudnn.flags(enabled=False):
                    loss_asr = F.ctc_loss(
                        log_probs,
                        asr_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.model.audio_tower[0].config.audio_patch_token,
                        reduction='mean',
                        zero_infinity=True,
                    )
            else:
                loss_asr=0

            shift_labels = labels
            if "label_shift" in outputs.keys() and len(outputs["label_shift"]) >0:
                if outputs["label_extend"] != -1:
                    new_shift_labels = torch.full(size=(shift_labels.shape[0], outputs["label_extend"]+shift_labels.shape[1]), fill_value=IGNORE_TOKEN_ID, dtype=torch.long).to(shift_labels.device)
                    for batch in range(len(outputs["label_shift"])):
                        it_lable_shift = outputs["label_shift"][batch]
                        it_speech_pos = outputs["speech_pos"][batch]
                        prefix = 0
                        for i in range(len(it_lable_shift)):
                            if i == len(it_lable_shift) - 1:
                                length = shift_labels.shape[1] - it_speech_pos[i] #len(shift_labels[batch]) - it_speech_pos[i]
                            else:
                                length =  it_speech_pos[i + 1] - it_speech_pos[i]
                            prefix += it_lable_shift[i]
                            new_shift_labels[batch][it_speech_pos[i] + prefix: it_speech_pos[i] + length + prefix]= shift_labels[batch][it_speech_pos[i]:it_speech_pos[i]+length]
                    shift_labels = new_shift_labels
                else:
                    raise NotImplementedError

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = shift_labels[..., 1:].contiguous()
            #print(shift_labels[:,:50])

            #print(shift_labels[:,:150])
            loss_fct = CrossEntropyLoss()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
                    
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss + 0.3 * loss_asr 

        t2u_output = None
        if do_task != None and do_task != "skip":
            if do_task == "finetune_kd":
                text_start_index = []
                for batch in range(len(outputs["label_shift"])):
                    text_start_index.append(outputs["speech_pos"][batch][0] + outputs["label_shift"][batch][0]+assistant_after_audio_shifts[batch])

                t2u_embeds_output = self.unit_translator.insert_text_embedding(
                    input_ids=t2u_input_ids,
                    attention_mask=t2u_attention_mask,
                    inputs_embeds=None,
                    labels=t2u_labels,
                    text_labels=labels,
                    shift_text_labels=new_shift_labels,
                    shift_text_hidden_states=hidden_states,
                    unit_targets=unit_targets,
                    sub_lengths=sub_lengths,
                    text_start_index=text_start_index,
                    do_task=do_task,
                )

                vae_loss, t2u_inputs_embeds, unit_targets, t2u_attention_mask = t2u_embeds_output

                t2u_output = self.unit_translator(
                    input_ids=None,
                    attention_mask=t2u_attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=t2u_inputs_embeds,
                    use_cache=use_cache,
                    labels=unit_targets,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                t2u_embeds_output = self.unit_translator.insert_text_embedding(
                    input_ids=t2u_input_ids,
                    attention_mask=t2u_attention_mask,
                    inputs_embeds=None,
                    labels=t2u_labels,
                    text_labels=labels,
                    shift_text_labels=new_shift_labels,
                    shift_text_hidden_states=hidden_states,
                    do_task=do_task,
                )
                vae_loss, t2u_inputs_embeds = t2u_embeds_output

                t2u_output = self.unit_translator(
                    input_ids=None,
                    attention_mask=t2u_attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=t2u_inputs_embeds,
                    use_cache=use_cache,
                    labels=t2u_labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            t2u_loss = t2u_output[0]
            # print(do_task, t2u_loss, vae_loss)
            if vae_loss != None:
                target_scale = t2u_loss.item() * 0.2
                vae_loss_weight = target_scale / vae_loss.item() if vae_loss > target_scale else 1.0
                t2u_loss = t2u_loss + vae_loss_weight * vae_loss
                #print(vae_loss)

            if loss != None:            # S2T + T2U loss
                # ignore LLM loss
                # t2u_output["loss"] = t2u_loss
                # return t2u_output
                # original version
                assert do_task in ["finetune"]
                if loss.item() < 1.0:   # 1.7
                    loss = 0.2 * loss + t2u_loss * 2.0
                else:
                    loss = loss + t2u_loss
            else:
                assert do_task in ["pretrain", "finetune_kd"]
                t2u_output["loss"] = t2u_loss
                return t2u_output
        
        #return CausalLMOutputWithPast(
        #    loss=loss,
        #    logits=outputs["audio_features"],
        #)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        model_inputs.update({"audios": kwargs["audios"]} if "audios" in kwargs.keys() else {})
        model_inputs.update({"do_task": kwargs["do_task"]} if "do_task" in kwargs.keys() else {})
        model_inputs.update({"return_dict": kwargs["return_dict_in_generate"]} if "return_dict_in_generate" in kwargs.keys() else {})
        return model_inputs

AutoConfig.register("EchoX", EchoXConfig)
AutoModelForCausalLM.register(EchoXConfig, EchoXForCausalLM)