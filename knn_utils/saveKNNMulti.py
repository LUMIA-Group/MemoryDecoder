import os
import time
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils
import pyarrow as pa

from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from loguru import logger

class KNNSearchMulti:
    def __init__(self, 
                 dstore_path,
                 val_path,
                 index_path,
                 output_path,
                 model_path,
                 k=1024,
                 knn_temp=1.0,
                 probe=32,
                 batch_size=32,
                 knn_gpu=True,
                 ignore_first=False,
                 threshold=1e-10):
        
        self.dstore_path = dstore_path
        self.val_path = val_path
        self.index_path = index_path
        self.output_path = output_path
        self.model_path = model_path
        self.k = k
        self.knn_temp = knn_temp
        self.probe = probe
        self.batch_size = batch_size
        self.knn_gpu = knn_gpu
        self.ignore_first = ignore_first
        
        self.threshold = threshold
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.process_index = self.accelerator.local_process_index
        
        # Get vocab size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.vocab_size = len(tokenizer)
        logger.info(f"Vocab size: {self.vocab_size}")
        
        # Load FAISS index (each process loads it)
        self.reconstruct_index, self.index = self._load_faiss_index()
        
        # Create dataset and dataloader
        dataset = Dataset.from_file(self.dstore_path)
        # Set format to torch for proper tensor conversion
        dataset.set_format(type='torch', columns=['keys', 'vals'])
        
        if self.val_path is not None:
            # Load val_file using pickle
            with open(self.val_path, 'rb') as f:
                self.vals = pickle.load(f).to(self.device)
        else:
            self.vals = dataset['vals'].to(self.device)
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4
        )
        
        # Prepare dataloader with accelerator
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        # Initialize Arrow writer on main process
        if self.process_index == 0:
            self._setup_arrow_writer()
    
    def _load_faiss_index(self):
        """Load FAISS index and optionally move to GPU"""
        logger.info(f"Process {self.process_index}: Loading FAISS index from {self.index_path}")
        
        cpu_index = faiss.read_index(self.index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        cpu_index.nprobe = self.probe
        
        if self.knn_gpu and faiss.get_num_gpus() > 0:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_id = self.process_index % faiss.get_num_gpus()
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), gpu_id, cpu_index, co)
            
            logger.info(f"Process {self.process_index}: Moved index to all GPU with shard")
        
        cpu_index.make_direct_map()
        
        return cpu_index, gpu_index
    
    def _setup_arrow_writer(self):
        """Set up Arrow writer for streaming writes (main process only)"""
        if self.process_index == 0:
            # Define schema for output Arrow file
            fields = [
                pa.field('id_cnt', pa.int32()),
                pa.field('token_id', pa.list_(pa.int32())),
                pa.field('prob', pa.list_(pa.float16())),  # Changed to float16
                pa.field('label', pa.int32())
            ]
            schema = pa.schema(fields)
            
            # Create output directory if needed
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create Arrow file and writer
            self.arrow_file = pa.OSFile(self.output_path, 'wb')
            self.arrow_writer = pa.ipc.new_stream(self.arrow_file, schema)

    def get_knns(self, queries, ignore_first=False):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries.to(torch.float32), self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        
        # If we need to ignore the first nearest neighbor
        # There seems to be a bug of faiss 1.11.0 cuvs, that the searched result isn't sorted by distance, please use faiss 1.12.0 w/o cuvs instead
        if ignore_first:
            return dists[:,1:], knns[:,1:]
        else:
            return dists, knns
    
    def knns_to_probs(self, knns, neg_dists):
        """Compute kNN probability distribution following the reference implementation"""
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temp, dim=-1).to(torch.float32).to(self.device)

        vals_at_knns = self.vals[knns].to(self.device)  # (batch * time, k)
        knn_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs)  # (batch * time, vocab)
        
        knn_probs = F.normalize(knn_probs, p=1, dim=-1)
        
        return knn_probs
    
    def sparsify_distribution(self, knn_probs):
        """Extract sparse representation of distribution with probs > threshold"""
        batch_size = knn_probs.shape[0]
        
        id_cnt_list = []
        token_id_list = []
        prob_list = []
        
        for b in range(batch_size):
            # Find indices where probability > threshold
            valid_mask = knn_probs[b] > self.threshold
            valid_ids = torch.nonzero(valid_mask).squeeze(-1)
            valid_probs = knn_probs[b][valid_ids]
            
            # Sort by probability (descending)
            sorted_indices = torch.argsort(valid_probs, descending=True)
            sorted_ids = valid_ids[sorted_indices]
            sorted_probs = valid_probs[sorted_indices]
            
            id_cnt_list.append(len(sorted_ids))
            token_id_list.append(sorted_ids.to(self.device))
            prob_list.append(sorted_probs.to(self.device).to(torch.float16))  # Convert to float16
        
        return id_cnt_list, token_id_list, prob_list
    
    def _save_step_data(self, id_cnt, token_id, prob, label):
        """Save data for current step using streaming Arrow format"""
        # Convert id_cnt to tensor for gathering
        id_cnt_tensor = torch.tensor(id_cnt, device=self.device)
        
        # Step 1: Gather id_cnt and label first
        id_cnt_gathered = self.accelerator.gather_for_metrics(id_cnt_tensor)
        label_gathered = self.accelerator.gather_for_metrics(label)
        
        # Step 2: Get the largest value of id_cnt from gathered tensor
        max_tokens_global = torch.max(id_cnt_gathered).item()
        
        # Step 3: Pad the original token_id and prob lists to the largest id_cnt
        batch_size = id_cnt_tensor.shape[0]
        token_id_padded = torch.full((batch_size, max_tokens_global), -1, dtype=torch.long, device=self.device)
        prob_padded = torch.zeros((batch_size, max_tokens_global), dtype=torch.float16, device=self.device)
        
        for i in range(batch_size):
            valid_count = id_cnt[i]
            current_token_id = token_id[i]
            current_prob = prob[i]
            
            # For the last batch, there may be list longer than max_tokens_global since there are some remainder. Truncate them !
            if current_token_id.shape[0] > max_tokens_global:
                logger.debug(f"Truncating token_id and prob for batch {i} from {current_token_id.shape[0]} to {max_tokens_global}")
                valid_count = max_tokens_global
                current_token_id = current_token_id[:max_tokens_global]
                current_prob = current_prob[:max_tokens_global]
            
            token_id_padded[i, :valid_count] = current_token_id
            prob_padded[i, :valid_count] = current_prob
        
        # Step 4: Gather token_id_tensor and prob_tensor
        token_id_gathered_padded = self.accelerator.gather_for_metrics(token_id_padded)
        prob_gathered_padded = self.accelerator.gather_for_metrics(prob_padded)
        
        # Step 5: Unpad the result token_id_tensor and prob_tensor to original length list
        total_batch_size = id_cnt_gathered.shape[0]
        token_id_gathered = []
        prob_gathered = []
        
        for i in range(total_batch_size):
            valid_count = id_cnt_gathered[i].item()
            token_id_gathered.append(token_id_gathered_padded[i, :valid_count].cpu().numpy())
            prob_gathered.append(prob_gathered_padded[i, :valid_count].cpu().numpy())
        
        # Only main process writes to file
        if self.process_index == 0:
            id_cnt_np = id_cnt_gathered.cpu().numpy()
            label_np = label_gathered.cpu().numpy()
            
            logger.info(f"id_cnt_np shape: {id_cnt_np.shape} ; token_id_gathered length: {len(token_id_gathered)}; prob_gathered length: {len(prob_gathered)}; label_np shape: {label_np.shape}")
            
            # Create Arrow arrays
            id_cnt_array = pa.array(id_cnt_np, type=pa.int32())
            token_id_array = pa.array(token_id_gathered, type=pa.list_(pa.int32()))
            prob_array = pa.array(prob_gathered, type=pa.list_(pa.float16()))
            label_array = pa.array(label_np, type=pa.int32())
            
            # Create batch and write
            batch = pa.RecordBatch.from_arrays(
                [id_cnt_array, token_id_array, prob_array, label_array],
                ['id_cnt', 'token_id', 'prob', 'label']
            )
            self.arrow_writer.write_batch(batch)
        
        self.accelerator.wait_for_everyone()
    
    def process(self):
        """Main processing loop"""
        logger.info(f"Process {self.process_index}: Starting kNN search and processing")
        
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Process {self.process_index}")):
            keys = batch['keys'].to(torch.float16).to(self.device)
            vals = batch['vals'].to(torch.int32).to(self.device)
            
            # Perform kNN search
            dists, knns = self.get_knns(keys, self.ignore_first)
            neg_dists = -dists
            
            # Compute probability distribution
            knn_probs = self.knns_to_probs(knns, neg_dists)
            
            # Sparsify distribution
            id_cnt, token_id, prob = self.sparsify_distribution(knn_probs)
            
            # Save step data
            self._save_step_data(id_cnt, token_id, prob, vals)
        
        # Close Arrow writer on main process
        if self.process_index == 0:
            self.arrow_writer.close()
            self.arrow_file.close()
            logger.info(f"Finished writing to {self.output_path}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-GPU kNN Search and Distribution Processing')
    parser.add_argument('--dstore_path', type=str, required=True,
                        help='Path to input Arrow file with keys and vals')
    parser.add_argument('--val_path', type=str, required=False, default=None,
                        help='Path to input Arrow file with vals')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to FAISS index file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output Arrow file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model for tokenizer loading')
    parser.add_argument('--k', type=int, default=1024,
                        help='Number of nearest neighbors to search')
    parser.add_argument('--knn_temp', type=float, default=1.0,
                        help='Temperature for kNN probability computation')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Temperature for kNN probability computation')
    parser.add_argument('--probe', type=int, default=32,
                        help='Number of probes for FAISS index')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--knn_gpu', action='store_true',
                        help='Use GPU for FAISS search')
    parser.add_argument('--ignore_first', type=bool, default=False,
                        help='whether to ignore the nearest neighbor')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Create KNNSearchMulti instance
    knn_search = KNNSearchMulti(
        dstore_path=args.dstore_path,
        val_path=args.val_path,
        index_path=args.index_path,
        output_path=args.output_path,
        model_path=args.model_path,
        k=args.k,
        knn_temp=args.knn_temp,
        probe=args.probe,
        batch_size=args.batch_size,
        knn_gpu=args.knn_gpu,
        ignore_first=args.ignore_first,
        threshold=args.threshold
    )
    
    # Process the data
    knn_search.process()

if __name__ == "__main__":
    main()