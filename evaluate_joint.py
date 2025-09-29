# Evaluation script with accelerate for multi-GPU perplexity evaluation

import sys
import inspect
import pdb 
import argparse
import json
import logging
import math
import time
import os
import random
from itertools import chain
import numpy as np
from pathlib import Path
import pickle
import torch.nn as nn 
from datasets import Dataset
from transformers.pytorch_utils import Conv1D

import datasets
import torch
from functools import partial
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset,load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GPT2LMHeadModel,
    GPT2Config,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch.nn.functional as F
from loguru import logger

# Import KNN utils
from knn_utils.saveEmbedMulti import KNNSaverMulti, KNNWrapperMulti, KEY_TYPE, DIST

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    
    # Train arguments
    
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="The block size for the model.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="The name of the project.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The name of the run.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Logging steps",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    
    # KNN arguments
    
    # KNN-LM args
    parser.add_argument("--knn_keytype", type=KEY_TYPE.from_string, default=KEY_TYPE.last_ffn_input, 
                        help="Key type for KNN")
    parser.add_argument("--lmbda", type=float, default=0.25,
                        help="Interpolation parameter lambda")
    parser.add_argument("--knn_temp", type=float, default=1.0,
                        help="Temperature for KNN")
    
    # Neural-KNN args
    parser.add_argument("--use_neural_knn", action="store_true", help="Enable Neural KNN")
    parser.add_argument("--knn_generator_path", type=str,
                        default=None,
                        help="Path to knn generator model")
    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    accelerator = Accelerator(**accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        log_level = logging.INFO
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        log_level = logging.ERROR

    logger.remove()
    logger.add(sys.stdout, format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <blue>{process.name}</blue> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level=log_level)

    # Intercept default logging and transform to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists.
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message.
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=log_level, force=True)
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.add_handler(InterceptHandler())
    logger.info(f"{accelerator.state}")

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------------------------------Load Dataset and Model------------------------------------------------------------
    # Load config and model
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    
    # -------------------------------------------------Load Dataset------------------------------------------------------------
    # Default to use gpt tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    lm_datasets = load_from_disk(args.dataset_name)[args.dataset_split_name]

    # Modify model embedding size
    vocab_size = len(tokenizer)
    model.resize_token_embeddings(vocab_size)
    
    # -------------------------------------------------Inject KNN------------------------------------------------------------

    dimension = model.config.hidden_size
    if os.path.exists(f"{args.knn_generator_path}/pytorch_model.bin"):
        knn_generator = AutoModelForCausalLM.from_pretrained(args.knn_generator_path, use_safetensors=False).to(model.device)
    else:
        knn_generator = AutoModelForCausalLM.from_pretrained(args.knn_generator_path, use_safetensors=True).to(model.device)
    
    knn_generator.resize_token_embeddings(vocab_size)

    # --------------------------------------------------Evaluation-----------------------------------------------------------
    
    def interpolate(knn_log_probs, lm_log_probs, lmbda=0.25):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), 
            knn_log_probs + np.log(lmbda))

        return interpolated

    def joint_evaluate(logits, knn_logits, batch, tokenizer, args):
        shift_logits = logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
        shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
        shift_knn_logits = knn_logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
        
        nonpad_mask = shift_labels != -100
        shift_logits = shift_logits[nonpad_mask] # (nonpad b*t, vocab_size)
        shift_knn_logits = shift_knn_logits[nonpad_mask] # (nonpad b*t, vocab_size)
        shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
        
        # Compute the entropy of the logits (shift_logits) and label_probs
        lm_log_probs = F.log_softmax(shift_logits, dim=-1)
        knn_log_probs = F.log_softmax(shift_knn_logits, dim=-1)
        interpolated_log_probs = interpolate(knn_log_probs, lm_log_probs, lmbda=args.lmbda)
        
        lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')
        joint_loss = F.nll_loss(interpolated_log_probs, shift_labels, reduction='sum')
        
        return joint_loss, lm_loss, shift_labels.shape[0]

    if args.do_test:
        eval_dataloader = DataLoader(
            lm_datasets, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        # Create a partial function with your specific parameters
        model, knn_generator ,eval_dataloader = accelerator.prepare( model, knn_generator, eval_dataloader )

        model.eval()
        knn_generator.eval()
        eval_lm = 0
        eval_joint = 0
        token_cnt = 0

        for batch in tqdm(eval_dataloader, desc = "Evaluating Model"):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None
                )
                knn_outputs = knn_generator(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None
                )
                
                joint_loss, lm_loss, cnt = joint_evaluate(outputs.logits, knn_outputs.logits, batch, tokenizer, args)
                eval_joint += joint_loss.item()
                eval_lm += lm_loss.item()
                token_cnt += cnt
                
        eval_lm = torch.tensor(eval_lm).to(model.device)
        eval_joint = torch.tensor(eval_joint).to(model.device)
        token_cnt = torch.tensor(token_cnt).to(model.device)
        
        eval_lm = accelerator.gather(eval_lm).sum().item()
        eval_joint = accelerator.gather(eval_joint).sum().item()
        token_cnt = accelerator.gather(token_cnt).sum().item()

        eval_lm = math.exp(eval_lm / token_cnt)
        eval_joint = math.exp(eval_joint / token_cnt)
        logger.info(f"token count: {token_cnt}")
        
        if accelerator.is_main_process:
            print(f"lm perplexity: {eval_lm}")
            print(f"joint perplexity: {eval_joint}")

if __name__ == "__main__":
    main()