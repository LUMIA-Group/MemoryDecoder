# <img src="assets/logo.png" alt="MemoryDecoder" width="60" height="60" style="vertical-align: middle"> Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2508.09874-b31b1b.svg)](https://www.arxiv.org/abs/2508.09874)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-MemoryDecoder-yellow)](https://huggingface.co/Clover-Hill/MemoryDecoder-gpt2-small)
[![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-Poster-blue)]()

</div>

<p align="center" style="font-size: larger;">
  <a href="https://www.arxiv.org/abs/2508.09874">Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models</a>
</p>

<p align="center">
  <strong>NeurIPS 2025 Poster</strong>
</p>

<p align="center">
<img src="assets/pipeline.png" width="95%">
</p>

## Overview

Memory Decoder introduces a novel paradigm for domain adaptation that bridges the gap between non-parametric retrieval methods and parametric fine-tuning approaches. By pre-training a compact transformer decoder to internalize retrieval patterns, Memory Decoder provides the benefits of both worlds:

- ✨ **Plug-and-Play**: A single Memory Decoder enhances any model sharing the same tokenizer
- 🚀 **Efficient Inference**: No retrieval overhead - just parallel forward passes  
- 🎯 **Domain Expertise**: Captures long-tail knowledge like non-parametric methods
- 🔒 **Preserves Capabilities**: Original model parameters remain unchanged

Unlike traditional approaches that either require expensive retraining (DAPT) or introduce significant inference latency (RAG), Memory Decoder offers efficient domain adaptation through a pretrained memory component that seamlessly integrates with existing models.

## 🚀 Quick Start

### Environment Setup

We run on **CUDA 12.4** with the following core dependencies:
- **faiss-gpu 1.11.0** (with cuvs support)
- **PyTorch 2.6.0**

#### Step 1: Install FAISS-GPU
```bash
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0
```

#### Step 2: Install PyTorch
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

#### Step 3: Install Other Dependencies
```bash
pip install transformers datasets accelerate pyarrow evaluate loguru wandb tqdm pickle
```

### Evaluate and Use Memory Decoder

We provide the checkpoint of gpt2-small Memory Decoder used in our experiments [🤗 gpt2-small Memory Decoder](https://huggingface.co/Clover-Hill/MemoryDecoder-gpt2-small). Simply download the checkpoint from huggingface and run the following scrtips:

#### Data Preprocessing
```bash
# scripts/preprocess_dataset.sh
TOKENIZER="/path/to/tokenizer(model)/directory"
OUTPUT_DIR=./dataset/wikitext-gpt2

python utils/preprocess_dataset.py \
    --dataset_name /path/to/wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --tokenizer_path ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --num_proc 32
```

#### Evaluate Base Model
```bash
# scripts/evaluate_base_gpt.sh
DATASET=/path/to/dataset
MODEL=/path/to/base/model
OUTPUT_DIR=tmp/

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python \
    -m train_base \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --eval_subset test \
    --output_dir ${OUTPUT_DIR} \
    --report_to none
```

#### Evaluate with Memory Decoder
```bash
# scripts/evaluate_joint_gpt2.sh
DATASET=/path/to/dataset
MODEL=/path/to/base/model
KNN_PATH=/path/to/memory/decoder
OUTPUT_DIR=tmp/

python -m evaluate_joint \
    --do_test \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name test \
    --per_device_eval_batch_size 16 \
    --output_dir ${OUTPUT_DIR} \
    --knn_temp 1 \
    --lmbda 0.55 \
    --knn_generator_path ${KNN_PATH} \
    --report_to none
```

### Performance Results on WikiText-103

|   Model    | Base | +MemDec | PPL Decrease |
|:----------:|:----:|:-------:|:-----------:|
| GPT2-small | 24.89 | **13.36** | -46.4% |
| GPT2-medium | 18.29 | **12.25** | -33.0% |
| GPT2-large | 15.80 | **11.53** | -27.0% |
| GPT2-xl | 14.39 | **10.93** | -24.0% |

### Generation Example

```python
# demo/generation_example.py
from memDec import MemoryDecoder
import transformers
from transformers import AutoModelForCausalLM
from loguru import logger

base_lm_path = "/fs-computility/plm/shared/jqcao/models/gpt2/gpt2-xl"
knn_generator_path = "/fs-computility/plm/shared/jqcao/projects/MemoryDecoder/checkpoint/memdec-gpt2-small"

tokenizer = transformers.AutoTokenizer.from_pretrained(base_lm_path)
base_lm = AutoModelForCausalLM.from_pretrained(base_lm_path)
knn_generator = AutoModelForCausalLM.from_pretrained(knn_generator_path)

base_lm.resize_token_embeddings(len(tokenizer))
knn_generator.resize_token_embeddings(len(tokenizer))
base_lm.eval()
knn_generator.eval()

joint = MemoryDecoder(base_lm, knn_generator, lmbda=0.55, knn_temp=1.0).to("cuda")

prompt = f"As with previous Valkyira Chronicles games , Valkyria Chronicles III is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

out_ids = joint.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Memory Decoder output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected: As with previous Valkyira Chronicles games , Valkyria Chronicles III is a role @-@ playing 
# video game developed by Sega and published by Sega for the PlayStation 2 .

out_ids = base_lm.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Base Model output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected: As with previous Valkyira Chronicles games , Valkyria Chronicles III is a turn-based 
# strategy game. The player takes control of a squad of Valkyria soldiers,
```

## 🛠️ Training Memory Decoder

### Training Pipeline

#### 1. Preprocess Dataset
Tokenize and group text for efficient processing:
```bash
bash scripts/preprocess_dataset.sh
```

#### 2. Build KNN Training Signals

Three-step process for creating supervision signals:

##### (1) Save Embeddings
```bash
python knn_utils/saveEmbedMulti.py
```

##### (2) Build IVFPQ Index
```bash
python knn_utils/build_index.py
```

##### (3) Search KNN Distributions
```bash
python knn_utils/saveKNNMulti.py
```

The complete pipeline is available in:
```bash
bash scripts/save_pipeline.sh
```

**Note:** Both embedding saving and KNN distribution search support multi-card multi-node inference/searching. Configure your `accelerate` settings appropriately for optimal performance.

#### 3. Start Training

Launch Memory Decoder training:
```bash
bash scripts/train_memdec.sh
```

The training interface is implemented in `train_memdec.py`.

## Citation

If you find Memory Decoder helpful in your research, please consider citing:

```bibtex
@article{cao2025memory,
  title={Memory decoder: A pretrained, plug-and-play memory for large language models},
  author={Cao, Jiaqi and Wang, Jiarui and Wei, Rubin and Guo, Qipeng and Chen, Kai and Zhou, Bowen and Lin, Zhouhan},
  journal={arXiv preprint arXiv:2508.09874},
  year={2025}
}
```

## Contact

For questions and discussions, please email: **maximus.cao@outlook.com**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.