from fairseq.dataclass.configs import FairseqConfig
from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.distributed import utils as distributed_utils
import torch
import json
from tqdm import tqdm
import random
import soundfile as sf
import numpy as np
import ast
import time
import math
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from collections import namedtuple
import sys
from argparse import Namespace
import argparse
import sentencepiece as spm
import re

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )

def tokenize(inputs, sp):
    text = re.sub(r'[^\w\s]', '', inputs.lower())
    inputs = ' '.join(sp.EncodeAsPieces(text))
    # print(inputs)
    return inputs

def get_t2u_config(model, beam=5):

    sys.argv = [
        "fairseq-interactive",
        "libri_t2u", 
        "--path", model, 
        "--gen-subset", "valid", 
        "--max-len-b", "1024", 
        "--max-source-positions", "500", 
        "--max-target-positions", "1024", 
        "--beam", str(beam), 
        "--results-path", "decode" 
    ]

    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    # distributed_utils.call_main(convert_namespace_to_omegaconf(args), load_text2units_model)
    return convert_namespace_to_omegaconf(args)

def load_text2units_model(cfg: FairseqConfig, device):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    return {
        "models": models, 
        "generator": generator, 
        "tokenizer": tokenizer, 
        "bpe": bpe, 
        "task": task, 
        "src_dict": src_dict,
        "tgt_dict": tgt_dict,
        "use_cuda": use_cuda
    }

def gen_units(model, cfg, inputs):
    inputs = [inputs]
    
    models = model['models']
    generator = model['generator']
    tokenizer = model['tokenizer']
    bpe = model['bpe']
    task = model['task']
    src_dict = model['src_dict']
    tgt_dict = model['tgt_dict']
    use_cuda = model['use_cuda']

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    start_id = 0
    results = []
    for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
        print("[INFO_DEBUG]", batch)
        bsz = batch.src_tokens.size(0)
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        constraints = batch.constraints
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            if constraints is not None:
                constraints = constraints.cuda()

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }
        translate_start_time = time.time()
        translations = task.inference_step(
            generator, models, sample, constraints=constraints
        )
        translate_time = time.time() - translate_start_time
        list_constraints = [[] for _ in range(bsz)]
        if cfg.generation.constraints:
            list_constraints = [unpack_constraints(c) for c in constraints]
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            constraints = list_constraints[i]
            results.append(
                (
                    start_id + id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                        "time": translate_time / len(translations),
                    },
                )
            )

    # print(results)

    units = []
    for id_, _, hypos, info in sorted(results, key=lambda x: x[0]):
        print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))

        # Process top predictions
        for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str="",
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )

        units.append(list(map(int, hypo_str.split(' '))))

    return units

def get_vocoder_config(vocoder, config):
    
    args = argparse.Namespace(
        vocoder=vocoder,
        vocoder_cfg=config,
        dur_prediction=True,  
        speaker_id=1,
        cpu=False  
    )

    return args

def load_units_vocoder(args, device):
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg).to(device)

    multispkr = vocoder.model.multispkr
    if multispkr:
        num_speakers = vocoder_cfg.get(
            "num_speakers", 200
        )  # following the default in codehifigan to set to 200
        assert (
            args.speaker_id < num_speakers
        ), f"invalid --speaker-id ({args.speaker_id}) with total #speakers = {num_speakers}"

    return vocoder, num_speakers if multispkr else 1, 'cuda' in device

def gen_wav(vocoder, args, data, device):
    vocoder, num_speakers, use_cuda = vocoder
    res = []
    for i, d in enumerate(data): # tqdm is removed for cleaner streaming
        x = {
            "code": torch.LongTensor(d).view(1, -1).to(device),
        }
        suffix = ""

        multispkr = vocoder.model.multispkr
        if multispkr:
            spk = (
                random.randint(0, num_speakers - 1)
                if args.speaker_id == -1
                else args.speaker_id
            )
            suffix = f"_spk{spk}"
            x["spkr"] = torch.LongTensor([spk]).view(1, 1)

        x = utils.move_to_cuda(x) if use_cuda else x
        wav = vocoder(x, args.dur_prediction).detach().cpu().numpy()
        
        res.append(wav)

    return res[0]