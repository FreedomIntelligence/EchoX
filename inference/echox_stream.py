import sys
from models.acllama_el_s2s import ACLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, WhisperProcessor
from peft import PeftModel, PeftConfig
import json
from tqdm import tqdm
import torch
import re
import os
torch.backends.cudnn.benchmark = False
import librosa
from .text_to_speech import *
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub import hf_hub_download
from typing import Dict, Optional, List
import tempfile
import select
from copy import deepcopy
from typing import Generator, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_model(args, device):
    quantization_config = None
    hf_token = os.getenv("HF_TOKEN")
    
    # load based model
    model = ACLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=None,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        token=hf_token,
    ).eval().to(device)
    for module in model.model.audio_tower:
        module = module.to(device)

    if args.peft_model_id:
        lora_config = PeftConfig.from_pretrained(args.peft_model_id)
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(model, args.peft_model_id, config=lora_config).to(
            dtype=torch.float16, device=device
        )
        model = model.merge_and_unload()

    model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, token=hf_token)

    audio_config = model.get_model().audio_tower[0].config
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    audio_config.llm_pad_token_id = tokenizer.pad_token_id
    audio_config.audio_patch_size = args.audio_token_len


    # whisper processor
    audio_processor = WhisperProcessor.from_pretrained(args.audio_tower, torch_dtype=torch.float16)

    # t2u
    unit_translator = model.get_unit_translator().eval()
    return model, audio_processor, tokenizer, unit_translator

def load_speech_model(device):
    vocoder = "./EchoX-Vocoder/g_00500000"
    vocoder_cfg = "./EchoX-Vocoder/config.json"
    voc_cfg = get_vocoder_config(vocoder, vocoder_cfg)
    vocoder = load_units_vocoder(voc_cfg, device)
    return vocoder, voc_cfg

class EchoxAssistant():
    def __init__(self):
        class BasicSetting:
            def __init__(self):
                self.device = "cuda:0"
                self.sampling_rate = 16000
                self.audio_token_len = 1  # 1500 = 300 token x 5 compress
                self.stop = "</s>"
                self.base_model_path = "./EchoX-8B"
                self.peft_model_id = None
                self.audio_tower = "./whisper-large-v3"
        self.args = BasicSetting()
        self.device = "cuda"
        self.vocoder, self.voc_cfg= load_speech_model(self.device)
        self.model, self.audio_processor, self.tokenizer, self.unit_translator = load_model(self.args, self.device)
        self.audio_executor = ThreadPoolExecutor(max_workers=2)
        # self.specAug = SpecAugmentTransform()
        # special_token
        DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
        audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * self.args.audio_token_len
        audio_placeholder = "\n"+audio_placeholder
        self.audio_placeholder_ids = self.tokenizer(audio_placeholder).input_ids

        self.begin_of_text_id = self.tokenizer.get_vocab()["<|begin_of_text|>"]
        self.start_header_id = self.tokenizer.get_vocab()["<|start_header_id|>"]
        self.end_header_id = self.tokenizer.get_vocab()["<|end_header_id|>"]
        self.eot_id = self.tokenizer.get_vocab()["<|eot_id|>"]
        self.nl_tokens = self.tokenizer('\n').input_ids
        self._system = self.tokenizer('system').input_ids
        self._user = self.tokenizer('user').input_ids
        self._assistant = self.tokenizer('assistant').input_ids
        self._speaker = self.tokenizer('speaker').input_ids

        self.max_len = 1024
        self.unit_max_len = 2048
        self.system_message = "You are a helpful language and speech assistant. \
You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."
    
    def _generate_audio_segment(self, segment_hidden_states):
        try:
            audio_units = self._generate_audio_units_from_hidden_states(segment_hidden_states)
            if audio_units:
                audio_float32 = self.generate_with_speech_model([list(map(int, audio_units.split(" ")))])
                audio_int16 = (audio_float32 * 32767).astype(np.int16)

                return (16000, audio_int16)
            return None
        except Exception as e:
            print(f"Background audio generation error: {e}")
            return None

    def gen_model_inputs(
            self,
            sources,
            tokenizer,
            max_len,
            system_message,
            audio_placeholder_ids, begin_of_text_id, start_header_id, end_header_id, eot_id, nl_tokens, _system, _user, _assistant,
        ) -> dict:
        # max_len 512

        # Apply prompt templates
        input_ids, audio_paths = [], []
        audio_path = []
        
        for source in sources:
            input_id = []
            system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
            input_id += system

            for j, item in enumerate(source["conversations"]):
                role = item["from"]
                value = item["value"]
                _audio_path = None

                if role == 'user':
                    if "audio" in item.keys():
                        _input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(value).input_ids + [eot_id]
                        _audio_path = item["audio"]
                    else:
                        _input_id = [start_header_id] + _user + [end_header_id] + tokenizer(value).input_ids + [eot_id]
                
                elif role == 'assistant':
                    _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                
                else:
                    raise NotImplementedError
                input_id += _input_id

                if _audio_path:
                    audio_path.append(_audio_path)
            assistant_input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens
            input_id += assistant_input_id

            audio_num = int(input_id.count(audio_placeholder_ids[-1]) / self.args.audio_token_len)
            assert len(audio_path) == audio_num
            if len(input_id) >= max_len:
                print(f"[WARNING] Your Input Length More Than {max_len}")
            input_ids.append(input_id[:max_len])
            audio_paths.append(audio_path)
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        return dict(
            input_ids=input_ids,
            audio_paths=audio_paths,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    def get_unit_result(self, ret):
        self.unit_translator.generation_config.pad_token_id = self.tokenizer.eos_token_id
        input_ids = ret["input_ids"]
        ret["input_ids"] = None
        model_outputs = self.unit_translator.generate(
            **ret,
            max_new_tokens=2048,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = model_outputs
        unit_output = self.tokenizer.batch_decode(output_ids)[0]
        if "▁" in unit_output:
            unit_output = ''.join(re.findall(r"<\|unit_(.*?)\|>", unit_output))
            
        units = re.findall(r'\d+', unit_output)

        #TODO grid of unk unit
        new_units = []
        for unit in units:
            if int(unit) < 1000:
                new_units.append(unit)

        units = ' '.join(new_units)
        return units
    

    def _inference(
        self,
        prompt,
        **kwargs,
    ):
        audio_paths = []
        response = []
        for item in prompt:
            for conv in item["conversations"]:
                if "audio" in conv:
                    audio_paths.append(conv["audio"])

        model_inputs = self.gen_model_inputs(
            prompt,
            self.tokenizer,
            self.max_len,
            self.system_message,
            self.audio_placeholder_ids,
            self.begin_of_text_id,
            self.start_header_id,
            self.end_header_id,
            self.eot_id,
            self.nl_tokens,
            self._system,
            self._user,
            self._assistant)
        
        audio_list = []
        if audio_paths and audio_paths[0] is not None:
            for audio_path in audio_paths:
                audio, _ = librosa.load(audio_path, sr=self.args.sampling_rate)
                audio_feat = self.audio_processor(audio, sampling_rate=self.args.sampling_rate, return_tensors="pt").input_features
                audio_list.append(audio_feat)
            audio_feats = torch.stack(audio_list, dim=0)
            audio_feats = audio_feats.to(dtype=torch.float16).to(self.device)
        
        if not audio_list:
            ret = dict(
                    input_ids=model_inputs["input_ids"].to(self.device),
                    attention_mask=model_inputs["attention_mask"].to(self.device),
                )
        else:
            ret = dict(
                    input_ids=model_inputs["input_ids"].to(self.device),
                    attention_mask=model_inputs["attention_mask"].to(self.device),
                    audios=audio_feats,
                )    
        
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        dot_input_ids = self.tokenizer(".", return_tensors="pt").input_ids.to(self.device) # shape: (1, 2), value: [[128000, 13]]
        period_token_id = dot_input_ids[0, -1]
        period_lm_head_embedding = self.model.lm_head.weight[period_token_id]

        input_ids = ret["input_ids"]
        attention_mask = ret["attention_mask"]
        input_token_len = input_ids.shape[1]

        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.2)
        top_p = kwargs.get('top_p', 0.9)
        do_sample = kwargs.get('do_sample', True)

        current_text = ""
        accumulated_hidden_states = []
        accumulated_tokens = []
        similarity_scores = []
        segment_start_idx = 0
        
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        past_key_values = None
        
        audio_futures = []
        segmentation_latency = 5

        with torch.no_grad():
            for step in range(max_new_tokens):
                while audio_futures and audio_futures[0].done():
                    completed_future = audio_futures.pop(0)
                    audio_data = completed_future.result()
                    if audio_data:
                        yield None, audio_data
                
                if current_input_ids is None:
                    break

                model_kwargs = {
                    "input_ids": current_input_ids,
                    "attention_mask": current_attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": True,
                    "do_task": "skip"
                }
                
                if step == 0 and "audios" in ret:
                    model_kwargs["audios"] = ret["audios"]
                
                outputs = self.model(**model_kwargs)
                
                logits = outputs.logits
                hidden_states = outputs.hidden_states[-1]
                past_key_values = outputs.past_key_values
                
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                if do_sample:
                    next_token_logits = next_token_logits / temperature
                    
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    current_input_ids = None
                    continue

                accumulated_tokens.append(next_token.item())
                last_hidden_state = hidden_states[0, -1]  # [hidden_dim]
                accumulated_hidden_states.append(last_hidden_state)
                
                similarity = F.cosine_similarity(last_hidden_state, period_lm_head_embedding, dim=0).item()
                similarity_scores.append(similarity)
                
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                current_text += token_text
                
                yield current_text, None
                
                current_idx = len(similarity_scores) - 1
                check_idx = current_idx - segmentation_latency
                if check_idx >= 0:
                    similarity_at_check = similarity_scores[check_idx]
                    is_peak = self._is_local_maximum(similarity_scores, check_idx, window=segmentation_latency)
                    should_segment = (is_peak and
                        check_idx - segment_start_idx >= 50) or (
                        is_peak and
                        similarity_at_check > 0.1 and
                        check_idx - segment_start_idx >= 20
                    )
                    
                    if should_segment:
                        segment_end_idx = check_idx + 1
                        
                        segment_hidden_states = torch.stack(
                            accumulated_hidden_states[segment_start_idx:segment_end_idx], dim=0
                        ).unsqueeze(0)
                        
                        future = self.audio_executor.submit(self._generate_audio_segment, segment_hidden_states)
                        audio_futures.append(future)
                        
                        segment_start_idx = segment_end_idx
                
                current_input_ids = next_token
                current_attention_mask = torch.ones_like(next_token)
            
            if segment_start_idx < len(accumulated_hidden_states):
                segment_hidden_states = torch.stack(
                    accumulated_hidden_states[segment_start_idx:], dim=0
                ).unsqueeze(0)
                future = self.audio_executor.submit(self._generate_audio_segment, segment_hidden_states)
                audio_futures.append(future)

            for future in audio_futures:
                audio_data = future.result()
                if audio_data:
                    yield None, audio_data

    def _is_local_maximum(self, scores, idx, window=5):
        start = max(0, idx - window)
        end = min(len(scores), idx + window + 1)
        local_scores = scores[start:end]
        return scores[idx] == max(local_scores)

    def _generate_audio_units_from_hidden_states(self, hidden_states):
        try:
            _, adapted_inputs_embeds = self.unit_translator.insert_text_embedding(
                inputs_embeds=hidden_states,
                do_task="skip",
            )
            
            attention_mask = torch.ones(adapted_inputs_embeds.shape[:2]).to(self.device)
            ret = dict(
                input_ids=None,
                inputs_embeds=adapted_inputs_embeds,
                attention_mask=attention_mask,
            )
            
            return self.get_unit_result(ret)
        except Exception as e:
            print(f"Error generating audio units: {e}")
            return None
        
    def generate_with_speech_model(self, units):
        wav = gen_wav(self.vocoder, self.voc_cfg, units, self.device)
        return wav