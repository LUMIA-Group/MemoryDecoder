#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

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

from utils.cal_loss import kl_loss_token, kl_loss_evaluate

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
    
    # KNN args
    
    # modify the following code to parser
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
        "--from_scratch",
        action="store_true",
    )
    parser.add_argument(
        "--init_lm_head",
        action="store_true",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    parser.add_argument(
        "--group_name",
        type=str,
    )
    parser.add_argument(
        "--lmbda", type=float, default=0.25,
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
    )
    parser.add_argument(
        "--knn_save_path", type=str, default=None,
    )
    
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

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    if args.report_to == "wandb":
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=args,
            init_kwargs={
                "wandb": {
                    "name": args.run_name if args.run_name is not None else None,
                    "group": args.group_name if args.group_name is not None else None,
                    "save_code": True,
                },
            }
        )

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

    if args.model_name_or_path and not args.from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
        
    # -------------------------------------------------Preprocess Dataset------------------------------------------------------------
    # Default to use gpt tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    lm_datasets = load_from_disk(args.dataset_name)[args.dataset_split_name]

    # Modify model embedding size
    vocab_size = len(tokenizer)
    model.resize_token_embeddings(vocab_size)

    # --------------------------------------------------Load Memmap-------------------------------------------------------------------
    
    # Load knn dstore and val file
    if accelerator.is_main_process:
        logger.info(f"Loading knn dstore and val dstore from {args.knn_save_path}...")
    knn_dstore = Dataset.from_file(args.knn_save_path)
    knn_dstore.set_format(type='torch', columns=['id_cnt', 'token_id', 'prob', 'label'])
    
    def knn_collate_fn(batch, knn_dstore, vocab_size):
        """
        Custom collate function that handles kNN data processing directly
        
        Args:
            batch: The batch of data
            knn_dstore: The KNN datastore
            vocab_size: Size of the vocabulary
            device: The device to move tensors to (accelerator.device)
        """
        # Apply default collation to the batch
        collated_batch = default_data_collator(batch)
        
        # Process kNN data for all items in the batch
        knn_labels_list = []
        knn_probs_list = []
        
        # Process each dstore_range in the batch
        for idx, cur_range in enumerate(collated_batch["dstore_range"]):
            # Get range boundaries
            start, end = int(cur_range[0]), int(cur_range[1])
            
            # Slice the knn_dstore
            knn_dstore_slice = knn_dstore.select(range(start, end))
            
            # Extract kNN label
            # Note that since datasets version 4.0.0, we can't use direct column selecting since the implementation of lazy columns, see pr https://github.com/huggingface/datasets/pull/7614
            cur_knn_label = knn_dstore_slice[:]["label"]
            knn_labels_list.append(cur_knn_label)
            
            # Extract token IDs and probabilities
            cur_token_id = knn_dstore_slice[:]["token_id"]
            cur_prob = knn_dstore_slice[:]["prob"]
            
            # Create sparse probability tensor
            cur_knn_prob = torch.zeros(size=(end - start, vocab_size))
            for i in range(end - start):
                assert cur_token_id[i].max() < vocab_size, f"token_id {cur_token_id[i]} is out of vocab size {vocab_size}"
                cur_knn_prob[i][cur_token_id[i]] = cur_prob[i]
            
            knn_probs_list.append(cur_knn_prob)
        
        # Concatenate and move to device
        collated_batch["knn_label"] = torch.cat(knn_labels_list, dim=0)
        collated_batch["knn_probs"] = torch.cat(knn_probs_list, dim=0)
        
        return collated_batch
    
    # --------------------------------------------------Evaluation-----------------------------------------------------------
    if args.do_test:
        collate_with_knn = partial(
            knn_collate_fn,
            knn_dstore=knn_dstore,
            vocab_size=vocab_size,
        )
        eval_dataloader = DataLoader(
            lm_datasets, collate_fn=collate_with_knn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
        # Create a partial function with your specific parameters
        model, eval_dataloader = accelerator.prepare( model, eval_dataloader )

        model.eval()
        eval_joint = 0
        eval_lm = 0
        total_token_num = 0

        for batch in tqdm(eval_dataloader, desc = "Evaluating Model"):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None
                )
                
                nll_loss, lm_loss, token_num = kl_loss_evaluate(outputs.logits, batch, tokenizer, args, batch["knn_label"], batch["knn_probs"])
                eval_joint += nll_loss.item()
                eval_lm += lm_loss.item()
                total_token_num += token_num

        eval_joint = torch.tensor(eval_joint).to(accelerator.device)
        eval_lm = torch.tensor(eval_lm).to(accelerator.device)
        total_token_num = torch.tensor(total_token_num).to(accelerator.device)
        
        eval_joint = accelerator.gather(eval_joint).sum().item()
        eval_lm = accelerator.gather(eval_lm).sum().item()
        total_token_num = accelerator.gather(total_token_num).sum().item()

        eval_joint = math.exp(eval_joint / total_token_num)
        eval_lm = math.exp(eval_lm / total_token_num)
        logger.info(f"joint perplexity: {eval_joint}")
        logger.info(f"lm perplexity: {eval_lm}")

        # Only do evaluation and no training
        return
        
    # --------------------------------------------------Start Training-----------------------------------------------------------
    train_dataset = lm_datasets
    eval_dataset = None
    
    # DataLoaders creation:
    collate_with_knn = partial(
        knn_collate_fn,
        knn_dstore=knn_dstore,
        vocab_size=vocab_size,
    )
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_with_knn, batch_size=args.per_device_train_batch_size, 
        shuffle=False,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=collate_with_knn, batch_size=args.per_device_eval_batch_size, 
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True
        )
    else:
        eval_dataloader = None

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)  / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader) 
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader) 

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    logging_interval_loss = 0
    logging_interval_kl_loss = 0
    logging_interval_lm_loss = 0
    total_loss = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None
                )
                
                loss, kl_loss, lm_loss = kl_loss_token(outputs.logits, batch, tokenizer, args, batch["knn_label"], batch["knn_probs"], alpha=args.alpha)

                # We keep track of the loss at each epoch
                logging_interval_loss += loss.detach().float()
                logging_interval_kl_loss += kl_loss.detach().float()
                logging_interval_lm_loss += lm_loss.detach().float()

                accelerator.backward(loss)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Clip gradnorm
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    actual_steps = args.gradient_accumulation_steps if not accelerator.gradient_state.end_of_dataloader else len(active_dataloader) % args.gradient_accumulation_steps

                    if actual_steps == 0:
                        actual_steps = args.gradient_accumulation_steps

                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / actual_steps / args.logging_steps
                    avg_kl_loss = accelerator.gather(logging_interval_kl_loss).mean().item() / actual_steps / args.logging_steps
                    avg_lm_loss = accelerator.gather(logging_interval_lm_loss).mean().item() / actual_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / actual_steps 

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "kl_loss": avg_kl_loss,
                        "lm_loss": avg_lm_loss,
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "rolling_loss":total_loss / completed_steps,
                    }
                    accelerator.log(to_be_logged,step=completed_steps)
                    if accelerator.is_main_process:
                        logger.info(f"step: {completed_steps}, loss: {avg_loss}, lr: {lr_scheduler.get_last_lr()[0]}")

                    logging_interval_loss = 0
                    logging_interval_kl_loss = 0
                    logging_interval_lm_loss = 0

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)

                    # Save state for continue training
                    accelerator.save_state(output_dir)

                    # Save unwrap model
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                          output_dir,
                          is_main_process=accelerator.is_main_process,
                          save_function=accelerator.save,
                          state_dict=accelerator.get_state_dict(model)
                    )

                    if accelerator.is_main_process:
                        unwrapped_model.config.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            # Save state for continue training
            accelerator.save_state(output_dir)

            # Save unwrap model
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                  output_dir,
                  is_main_process=accelerator.is_main_process,
                  save_function=accelerator.save,
                  state_dict=accelerator.get_state_dict(model)
            )

            if accelerator.is_main_process:
                unwrapped_model.config.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()