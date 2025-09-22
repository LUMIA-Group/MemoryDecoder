"""
Build a FAISS index from an Arrow file containing keys and values.
Only requires the path to the datastore Arrow file.
"""

import os
import torch
import multiprocessing
import time
from loguru import logger
import argparse
import pickle
import numpy as np
import faiss
import re
from datasets import Dataset
from tqdm import tqdm
from transformers import Qwen2Model

def parse_dstore_path(dstore_path):
    """
    Parse the dstore path to extract model_type, eval_subset, and dimension.
    
    Example:
    /fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/dstore_qwen2_train_3584.arrow
    """
    # Extract directory and filename
    dstore_dir = os.path.dirname(dstore_path)
    filename = os.path.basename(dstore_path)
    
    # Extract dimension from filename (the number before .arrow)
    dimension_match = re.search(r'_(\d+)\.arrow$', filename)
    if not dimension_match:
        raise ValueError(f"Could not extract dimension from filename: {filename}")
    dimension = int(dimension_match.group(1))
    
    # Extract eval_subset from filename (usually between model and dimension)
    # Format typically: dstore_model_subset_dimension.arrow
    parts = filename.split('_')
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    eval_subset = parts[-2]  # Second to last part (before dimension)
    
    # Get model_type from the parent directory structure
    # The parent directory of the dstore directory is typically the model name
    model_dir = os.path.basename(dstore_dir)
    parent_dir = os.path.basename(os.path.dirname(dstore_dir))
    model_type = parent_dir
    
    return {
        "dstore_dir": dstore_dir,
        "model_type": model_type,
        "eval_subset": eval_subset,
        "dimension": dimension
    }

def get_index_path(dstore_info):
    """Generate the path for the FAISS index file."""
    index_path = os.path.join(
        dstore_info["dstore_dir"], 
        f"{dstore_info['eval_subset']}_{dstore_info['dimension']}.index"
    )
    return index_path

def select_continuous_chunks(dataset, total_sample_size, num_chunks=10, seed=42):
    """
    Select continuous chunks of data from non-overlapping regions of the dataset.
    
    Args:
        dataset: The dataset to sample from
        total_sample_size: Total approximate number of samples to select
        num_chunks: Number of continuous chunks to select
        seed: Random seed for reproducibility
        
    Returns:
        Numpy array of selected samples
    """
    logger.info(f"Selecting ~{total_sample_size} samples in {num_chunks} continuous chunks")
    np.random.seed(seed)
    dataset_size = len(dataset)
    
    # Calculate samples per chunk (approximate)
    samples_per_chunk = max(1, total_sample_size // num_chunks)
    
    # Dataset is too small for chunking
    if dataset_size <= total_sample_size:
        logger.warning(f"Dataset size ({dataset_size}) is smaller than requested sample size, using all data")
        return np.array(dataset[:]['keys']).astype(np.float32)
    
    # Divide dataset into non-overlapping regions
    region_size = dataset_size // num_chunks
    
    all_samples = []
    total_selected = 0
    
    logger.info(f"Selecting {num_chunks} chunks with ~{samples_per_chunk} samples each")
    for i in range(num_chunks):
        # Calculate region boundaries
        region_start = i * region_size
        region_end = (i + 1) * region_size if i < num_chunks - 1 else dataset_size
        
        # Calculate valid starting range within this region
        # The starting point should allow the chunk to fit within the region
        max_start = max(region_start, region_end - samples_per_chunk - 1)
        
        # If the region is smaller than samples_per_chunk, use the whole region
        if max_start <= region_start:
            start_idx = region_start
            end_idx = region_end
        else:
            # Randomly select a starting point within valid range
            start_idx = np.random.randint(region_start, max_start + 1)
            end_idx = min(start_idx + samples_per_chunk, region_end)
        
        chunk_size = end_idx - start_idx
        
        logger.info(f"Chunk {i+1}/{num_chunks}: Region [{region_start}:{region_end}], Selected [{start_idx}:{end_idx}] ({chunk_size} samples)")
        chunk_data = np.array(dataset.select(range(start_idx, end_idx))['keys']).astype(np.float32)
        all_samples.append(chunk_data)
        total_selected += chunk_size
        
    # Concatenate all chunks
    logger.info(f"Selected {total_selected} samples in total across {num_chunks} non-overlapping chunks")
    return np.vstack(all_samples)

def build_index(
    dstore_path,
    num_keys_to_add_at_a_time=1_000_000,
    ncentroids=4096,
    seed=42,
    code_size=32,
    probe=8
):
    """
    Build a FAISS index from an Arrow file containing keys and values.
    
    Args:
        dstore_path: Path to the Arrow file containing the datastore
        num_keys_to_add_at_a_time: Number of keys to add at a time
        ncentroids: Number of centroids for IVFPQ
        seed: Random seed
        code_size: Code size for PQ
        probe: Number of probes for query
    """
    # Parse the dstore path to extract necessary information
    dstore_info = parse_dstore_path(dstore_path)
    logger.info(f"Parsed dstore path: {dstore_info}")
    
    dimension = dstore_info["dimension"]
    
    logger.info('Loading Dataset...')
    dstore = Dataset.from_file(dstore_path)
    # Set format to numpy for proper array conversion
    dstore.set_format(type='numpy', columns=['keys', 'vals'])
    
    # Save dstore.vals to a separate pickle file for later use
    vals_path = os.path.join(dstore_info["dstore_dir"], f"{dstore_info['eval_subset']}_vals.pkl")
    with open(vals_path, 'wb') as f:
        pickle.dump(torch.tensor(dstore['vals']), f)
    
    logger.info('Building index...')
    index_name = get_index_path(dstore_info)
    logger.info(f"Index will be saved to: {index_name}")

    max_threads = multiprocessing.cpu_count() 
    faiss.omp_set_num_threads(max_threads)
    logger.info(f'Total CPU count: {max_threads}')
    
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, ncentroids, code_size, 8)
    index.nprobe = probe
    
    logger.info('Training Index...')
    sample_size = min(1000000, len(dstore))
    train_data = select_continuous_chunks(dstore, total_sample_size=sample_size, num_chunks=1000, seed=seed)

    logger.info(f'End Selecting data. Start Training...')
    
    start = time.time()
    index.train(train_data)
    logger.info(f'Training took {time.time() - start:.2f} s')
    
    logger.info('Adding Keys...')
    start_time = time.time()
    
    for start in tqdm(range(0, len(dstore), num_keys_to_add_at_a_time)):
        end = min(len(dstore), start + num_keys_to_add_at_a_time)
        to_add = dstore.select(range(start,end))['keys'].astype(np.float32)  
        
        index.add_with_ids(to_add, np.arange(start, end))
    
    faiss.write_index(index, index_name)
    logger.info(f'Added {len(dstore)} keys in {time.time() - start_time:.2f} s')
    return index_name

def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from an Arrow datastore")
    
    # Only required parameter is the dstore_path
    parser.add_argument(
        "--dstore_path", 
        type=str, 
        required=True,
        help="Path to the Arrow file containing the datastore"
    )
    
    # Optional parameters
    parser.add_argument(
        "--num_keys_to_add_at_a_time", 
        type=int, 
        default=1_000_000,
        help="Number of keys to add at a time"
    )
    parser.add_argument(
        "--ncentroids", 
        type=int, 
        default=4096,
        help="Number of centroids for IVFPQ"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--code_size", 
        type=int, 
        default=32,
        help="Code size for PQ"
    )
    parser.add_argument(
        "--probe", 
        type=int, 
        default=8,
        help="Number of probes for query"
    )
    
    args = parser.parse_args()
    
    index_path = build_index(
        dstore_path=args.dstore_path,
        num_keys_to_add_at_a_time=args.num_keys_to_add_at_a_time,
        ncentroids=args.ncentroids,
        seed=args.seed,
        code_size=args.code_size,
        probe=args.probe
    )
    
    logger.info(f"Index saved to: {index_path}")

if __name__ == "__main__":
    main()