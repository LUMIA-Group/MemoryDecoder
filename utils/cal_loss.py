import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np
from loguru import logger
import os

EMBED_PAD_NUM = -10000

def interpolate(knn_log_probs, lm_log_probs, lmbda=0.25):
    interpolated = torch.logaddexp(
        lm_log_probs + np.log(1 - lmbda), 
        knn_log_probs + np.log(lmbda))

    return interpolated

def kl_loss_evaluate(logits, batch, tokenizer, args, knn_label, knn_prob):
    label_probs = knn_prob
    
    shift_logits = logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    
    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask] # (nonpad b*t, vocab_size)
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
    label_probs = label_probs / label_probs.sum(dim=-1, keepdim=True) # Normalize label_probs
    
    # Ensure that the dimensions match
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"
    assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"
    
    # Compute the label_probs
    shift_probs = F.softmax(shift_logits, dim=-1)

    # Calculate PPL
    label_log_probs = label_probs.log()
    label_log_probs = torch.nan_to_num(label_log_probs, nan=None, neginf=-10000.0)
    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    interpolate_log_probs = interpolate(label_log_probs, lm_log_probs, lmbda=args.lmbda)
    nll_loss = F.nll_loss(interpolate_log_probs, shift_labels, reduction='sum')
    lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')
    token_num = shift_labels.shape[0]
    
    return nll_loss, lm_loss, token_num

def kl_loss_token(logits, batch, tokenizer, args, knn_label, knn_prob, alpha=0.5):
    label_probs = knn_prob
    
    shift_logits = logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    
    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask] # (nonpad b*t, vocab_size)
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
    label_probs = label_probs / label_probs.sum(dim=-1, keepdim=True) # Normalize label_probs
    
    # Ensure that the dimensions match
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"
    assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"
    
    # MemDec loss
    kl_loss = F.kl_div(F.log_softmax(shift_logits, dim=-1), label_probs, reduction='batchmean')
    
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits, shift_labels)
    
    total_loss = alpha * kl_loss + (1 - alpha) * lm_loss
    
    logger.info(f"KL loss: {kl_loss} LM loss: {lm_loss} Total loss: {total_loss}")
    
    return total_loss, kl_loss, lm_loss