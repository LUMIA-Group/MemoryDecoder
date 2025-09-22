#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import pickle
import itertools
from loguru import logger
import logging
import inspect
import logging
import math
import os
    
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

import datasets
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from knn_utils.saveEmbedMulti import KNNSaverMulti, KNNWrapperMulti, KEY_TYPE, DIST

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    from_scratch: bool = field(
        default=False,
        metadata={
            "help": "Train the model from scratch, starting with a new config, model and tokenizer."
        },
    )
    group_name: str = field(default="default")

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    eval_subset: str = field(default='validation')
    stride: int = field(default=512)
    patience: int = field(default=None)
    prompt: str = field(default=None)

@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # KNN-LM args:
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=False)
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="/fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews")
    dstore_file: str = field(default="/fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/dstore_qwen2_train_3584.arrow")
    val_file: str = field(default="/fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/train_val.pickle")
    index_file: str = field(default="/fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/train_3584.index")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.25)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
    probe: int = field(default=32)

def main():
    # -----------------------------------------------------------Arguments-----------------------------------------------------------

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()

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
    datasets.utils.logging.set_verbosity(datasets.logging.WARNING)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.add_handler(InterceptHandler())

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"kNN parameters {knn_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # -----------------------------------------------------------Load Datasets-----------------------------------------------------------

    if data_args.dataset_name is not None:
        lm_datasets = load_from_disk(data_args.dataset_name)
    else:
        raise ValueError("Need to specify a dataset")

    if not (training_args.do_train or data_args.eval_subset == 'train'):
        # If not training and not evaluating on train, we do not need to process it
        del lm_datasets["train"]

    # -----------------------------------------------------------Load Models-----------------------------------------------------------
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if model_args.model_name_or_path and not model_args.from_scratch:
        if not os.path.exists(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")):
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                use_safetensors=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                use_safetensors=False
            )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # -----------------------------------------------------------Setup KNN-----------------------------------------------------------

    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = lm_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
    )

    if knn_args.knn:
        knn_wrapper = KNNWrapperMulti(val_file=knn_args.val_file, index_file=knn_args.index_file, dimension = dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype, knn_gpu=knn_args.knn_gpu,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe, local_process_index=training_args.local_process_index)

    elif knn_args.save_knnlm_dstore:
        knn_wrapper = KNNSaverMulti(dstore_dir=knn_args.dstore_dir, dimension=dimension, knn_keytype=knn_args.knn_keytype, knn_gpu=knn_args.knn_gpu, eval_subset=data_args.eval_subset, training_args = training_args, accelerator=trainer.accelerator)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)

    # -----------------------------------------------------------Start Training-----------------------------------------------------------
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"perplexity:{perplexity}")
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()

if __name__ == "__main__":
    main()