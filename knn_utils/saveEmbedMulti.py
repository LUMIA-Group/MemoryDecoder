import os

import torch.nn.functional as F
from copy import deepcopy
from loguru import logger
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
import pickle
from tqdm import tqdm

import contextlib
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing
from datasets import Dataset
from typing import List, Dict, Optional, Union, Tuple

import faiss
import faiss.contrib.torch_utils

class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()

class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapperMulti(object):
    def __init__(self, val_file, index_file, dimension, 
            knn_sim_func=None, knn_keytype=None, knn_gpu=True,
            k=1024, lmbda=0.25, knn_temp=1.0, probe=8, local_process_index=None):
        self.val_file = val_file
        self.index_file = index_file
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2
        self.knn_keytype = KEY_TYPE.last_ffn_input
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.local_process_index = local_process_index

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapperMulti.l2,
            DIST.dot: KNNWrapperMulti.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[self.knn_sim_func] # l2 or dot product function

    def setup_faiss(self):
        if not self.val_file:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        cpu_index = faiss.read_index(self.index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            # 1. Set maximum GPU memory allocation
            res = faiss.StandardGpuResources()
            res.setTempMemory(40 * 1024 * 1024 * 1024)

            # 2. Create a more memory-efficient cloner
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True           # Use half precision
            co.verbose = True              # See what's happening

            gpu_index = faiss.index_cpu_to_gpu(res, self.local_process_index, cpu_index, co)
            
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()

        start_time = time.time()
        # Load val_file using pickle
        with open(self.val_file, 'rb') as f:
            self.vals = pickle.load(f).to(self.device)
        logger.info(f'Loading keys and vals to memory took {time.time() - start_time} s')

        return cpu_index, gpu_index

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward

        # Create a wrapper that calls pre_forward_hook, ensuring that self still refers to KNNSaverMulti
        def forward_wrapper(input_ids=None, attention_mask=None, labels=None, **kwargs):
            return self.pre_forward_hook(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Override the model's forward with our wrapper
        model.forward = forward_wrapper
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapperMulti.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapperMulti.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features
        
    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries.to(torch.float32), self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        self.input_ids = input_ids
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask] # (nonpad, dim)
        
        dists, knns = self.get_knns(queries) # (nonpad batch * time, k)
        
        # Compute knn probs
        neg_dists = -dists
        knn_log_probs = self.knns_to_log_prob(knns, neg_dists) # (nonpad b*t, vocab_size)
        
        # Interpolate
        interpolated_scores = KNNWrapperMulti.interpolate(dists, knn_log_probs, lm_logits, self.lmbda) # (nonpad b * t, vocab)
        output[nonpad_mask] = interpolated_scores.to(output.dtype)
        
        return output 

    def knns_to_probs(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1).to(torch.float32)
        vals_at_knns = self.vals[knns].squeeze(-1).to(probs.device) # (nonpad batch * time, k)
        knn_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs) # (nonpad_batch * time, vocab)
        knn_probs = F.normalize(knn_probs, p=1, dim=-1)
        
        return knn_probs

    def knns_to_log_prob(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs
    
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
        
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)


    @staticmethod
    def interpolate(dists, knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), 
            knn_log_probs + np.log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte
        elif model_type == "qwen2":
            return lambda model: model.embed_tokens

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        },
        'llama': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.layers[-1], False),
        },
        'qwen2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.layers[-1], False),
        },
}

class KNNSaverMulti(object):
    def __init__(self, dstore_dir, dimension, knn_keytype=None, knn_gpu=False, training_args=None, eval_subset=None, accelerator=None):
        self.eval_subset = eval_subset
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.training_args = training_args
        
        # Multi-GPU settings
        self.world_size = training_args.world_size if training_args else 1
        self.process_index = training_args.local_process_index if training_args else 0
        self.device = training_args.device if training_args else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accelerator = accelerator
        
        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.hook_handles = []
        self.knn_gpu = knn_gpu
        
        if self.process_index ==0:
            if not os.path.exists(self.dstore_dir):
                logger.info(f"Creating directory {self.dstore_dir} for storing the datastore.")
                os.makedirs(self.dstore_dir, exist_ok=True)

    def _get_arrow_file_path(self):
        """Get the Arrow file path (single file for all processes)"""
        base_path = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.eval_subset, self.dimension)
        return base_path

    def _setup_arrow_writer(self):
        """Set up the Arrow writer for streaming writes (only on main process)"""
        if self.process_index == 0:
            logger.info(f"Setting up arrow writer...")
            # Define schema for the Arrow file
            fields = [
                pa.field('keys', pa.list_(pa.float16(), self.dimension)),
                pa.field('vals', pa.int32())
            ]
            schema = pa.schema(fields)
            
            # Create the Arrow file and writer using streaming format
            self.arrow_file = pa.OSFile(self.arrow_file_path, 'wb')
            self.arrow_writer = pa.ipc.new_stream(self.arrow_file, schema)

    def _save_step_data(self, keys, vals):
        # Detach from computation graph and ensure contiguous memory
        keys_tensor = keys.detach().contiguous()
        vals_tensor = vals.detach().contiguous()
        
        # Convert to desired types
        keys_tensor = keys_tensor.to(dtype=torch.float16)
        vals_tensor = vals_tensor.to(dtype=torch.int32)

        # Gather from all processes
        all_keys = self.accelerator.gather_for_metrics(keys_tensor)
        all_vals = self.accelerator.gather_for_metrics(vals_tensor)

        shift = 0 if self.is_encoder_decoder else 1
        if shift == 1:
            all_keys = all_keys[:, :-shift]
        all_keys = all_keys.flatten(0, 1)  # (batch * time, dim)
        all_vals = all_vals[:, shift:].flatten(0, 1)  # (batch * time)

        nonpad_mask = all_vals != -100
        all_keys = all_keys[nonpad_mask]
        all_vals = all_vals[nonpad_mask]
            
        # Only main process writes to file
        if self.process_index == 0:
            # Convert tensors back to numpy arrays
            all_keys_np = all_keys.cpu().numpy().astype(np.float16)
            all_vals_np = all_vals.cpu().numpy().astype(np.int32)
            
            # Create Arrow arrays
            keys_list = [all_keys_np[i] for i in range(all_keys_np.shape[0])]
            keys_array = pa.array(keys_list, type=pa.list_(pa.float16(), self.dimension))
            vals_array = pa.array(all_vals_np, type=pa.int32())
            
            # Create batch and write
            batch = pa.RecordBatch.from_arrays([keys_array, vals_array], ['keys', 'vals'])
            self.arrow_writer.write_batch(batch)
            
            logger.info(f"Main process: Flushed buffer, total saved: {self.dstore_idx + all_keys_np.shape[0]}")
            self.dstore_idx += all_keys_np.shape[0]

        self.accelerator.wait_for_everyone()
        
    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapperMulti.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward

        # Create a wrapper that calls pre_forward_hook
        def forward_wrapper(input_ids=None, attention_mask=None, labels=None, **kwargs):
            return self.pre_forward_hook(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Override the model's forward with our wrapper
        model.forward = forward_wrapper

        # Inject our main function after the model's final layer
        final_layer = KNNWrapperMulti.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        # Set up Arrow writer now that we have the model info
        self.arrow_file_path = self._get_arrow_file_path()
        if os.path.exists(self.arrow_file_path):
            logger.info(f'Arrow file already exists at {self.arrow_file_path}.')
        else:
            logger.info(f"Creating Arrow file at {self.arrow_file_path}.")
            self._setup_arrow_writer()

        # Create directory if it doesn't exist
        Path(self.dstore_dir).mkdir(parents=True, exist_ok=True)
        
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        captured_keys = self.activation_capturer.captured

        self._save_step_data(keys = captured_keys, vals = self.labels)
        
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        # Close the Arrow writer (main process only)
        if self.process_index == 0 and hasattr(self, 'arrow_writer'):
            self.arrow_writer.close()
            self.arrow_file.close()
        
        # Remove hooks
        for h in self.hook_handles:
            h.remove()
        
        if self.model is not None and hasattr(self.model, 'broken_into') and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
        
    def build_index(self, num_keys_to_add_at_a_time=1_000_000, ncentroids=4096, seed=42, code_size=32, probe=8):
        logger.info('Loading Dataset...')
        dstore_path = self._get_arrow_file_path()
        dstore = Dataset.from_file(dstore_path)
        # Set format to numpy for proper array conversion
        dstore.set_format(type='numpy', columns=['keys', 'vals'])
        
        logger.info('Building index...')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.eval_subset, self.dimension)
        
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension, ncentroids, code_size, 8)
        index.nprobe = probe
        
        logger.info('Training Index...')
        np.random.seed(seed)
        sample_size = min(200000, len(dstore))
        random_sample = np.random.choice(len(dstore), size=sample_size, replace=False)
        
        train_data = np.array(dstore.select(random_sample)['keys']).astype(np.float32)
        start = time.time()
        index.train(train_data)
        logger.info(f'Training took {time.time() - start:.2f} s')
        
        logger.info('Adding Keys...')
        start_time = time.time()
        
        for start in tqdm(range(0, len(dstore), num_keys_to_add_at_a_time)):
            end = min(len(dstore), start + num_keys_to_add_at_a_time)
            to_add = np.array(dstore.select(range(start,end))['keys']).astype(np.float32)  
            
            index.add_with_ids(to_add, np.arange(start, end))
            
            if (start // num_keys_to_add_at_a_time) % 10 == 0:
                faiss.write_index(index, index_name)
        
        faiss.write_index(index, index_name)
        logger.info(f'Added {len(dstore)} keys in {time.time() - start_time:.2f} s')

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, eval_subset, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{eval_subset}_{dimension}.arrow'

def get_index_path(dstore_dir, model_type, eval_subset, dimension):
    return f'{dstore_dir}/index_{model_type}_{eval_subset}_{dimension}.index'

def get_result_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/test_retrieve_result(no_recompute).pickle'