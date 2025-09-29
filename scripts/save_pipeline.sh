#!/bin/bash

# ========================================
# Memory Decoder Training Data Preparation Pipeline Configuration
# ========================================

# Model Configuration
MODEL_FAMILY="gpt2"
MODEL_SIZE="xl"
DIMENSION=1600

# Dataset Configuration
DATASET_NAME="wikitext"
SUBSET="train"

# Path Configuration
DATASET="/path/to/HF/local/arrow/dataset"
ACCELERATE_CONFIG="./accelerate_config/${MODEL_FAMILY}.yaml"
DSTORE_DIR="./dstore/${MODEL_FAMILY}-${MODEL_SIZE}/${DATASET_NAME}"
OUTPUT_DIR="./results/tmp/${MODEL_FAMILY}-${MODEL_SIZE}-${DATASET_NAME}-ppl"
# It is suggested to use the finetuned version of gpt-xl model in https://huggingface.co/Clover-Hill/gpt2-xl-finetuned-wikitext103
MODEL_TO_SAVE="/path/to/training/objective/building/model"

# Training Configuration
BATCH_SIZE_EVAL=32
BATCH_SIZE_KNN=16000

# KNN Configuration
K=1024
KNN_TEMP=16.0
PROBE=32
NCENTROIDS=4096
CODE_SIZE=64
NUM_KEYS_TO_ADD=10000000

# Derived paths
DSTORE_PATH="${DSTORE_DIR}/dstore_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}.arrow"
VAL_PATH="${DSTORE_DIR}/${SUBSET}_vals.pkl"
INDEX_PATH="${DSTORE_DIR}/${SUBSET}_${DIMENSION}.index"
OUTPUT_PATH="${DSTORE_DIR}/knn_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}.arrow"

# ========================================
# Pipeline Execution
# ========================================

echo "=========================================="
echo "NeuralKNN Pipeline"
echo "=========================================="
echo "Model: ${MODEL_FAMILY}-${MODEL_SIZE}"
echo "Dataset: ${DATASET_NAME}"
echo "Subset: ${SUBSET}"
echo "=========================================="
echo ""

# Step 1: Generate Datastore
echo "[Step 1/3] Generating datastore..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Datastore directory: ${DSTORE_DIR}"
echo ""

WANDB_PROJECT="neuralKNN" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train_base \
    --model_name_or_path ${MODEL_TO_SAVE} \
    --dataset_name ${DATASET} \
    --do_eval --eval_subset ${SUBSET} \
    --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
    --output_dir ${OUTPUT_DIR} \
    --dstore_dir ${DSTORE_DIR} \
    --save_knnlm_dstore \
    --report_to none

if [ $? -ne 0 ]; then
    echo "Error: Datastore generation failed!"
    exit 1
fi

echo ""
echo "[Step 1/3] ✓ Datastore generation completed"
echo ""

# Step 2: Build Index
echo "[Step 2/3] Building FAISS index..."
echo "Datastore path: ${DSTORE_PATH}"
echo "Number of centroids: ${NCENTROIDS}"
echo "Code size: ${CODE_SIZE}"
echo "Probe: ${PROBE}"
echo ""

python -m knn_utils.build_index \
    --dstore_path ${DSTORE_PATH} \
    --num_keys_to_add_at_a_time ${NUM_KEYS_TO_ADD} \
    --ncentroids ${NCENTROIDS} \
    --code_size ${CODE_SIZE} \
    --probe ${PROBE}

if [ $? -ne 0 ]; then
    echo "Error: Index building failed!"
    exit 1
fi

echo ""
echo "[Step 2/3] ✓ Index building completed"
echo ""

# Step 3: Save KNN Results
echo "[Step 3/3] Saving KNN results..."
echo "Model: ${MODEL_TO_SAVE}"
echo "K neighbors: ${K}"
echo "KNN temperature: ${KNN_TEMP}"
echo "Output path: ${OUTPUT_PATH}"
echo ""

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    knn_utils.saveKNNMulti \
    --model_path ${MODEL_TO_SAVE} \
    --dstore_path ${DSTORE_PATH} \
    --val_path ${VAL_PATH} \
    --index_path ${INDEX_PATH} \
    --output_path ${OUTPUT_PATH} \
    --k ${K} \
    --knn_temp ${KNN_TEMP} \
    --probe ${PROBE} \
    --batch_size ${BATCH_SIZE_KNN} \
    --ignore_first True \
    --knn_gpu

if [ $? -ne 0 ]; then
    echo "Error: KNN saving failed!"
    exit 1
fi

echo ""
echo "[Step 3/3] ✓ KNN results saved"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Final outputs:"
echo "  - Datastore: ${DSTORE_PATH}"
echo "  - Index: ${INDEX_PATH}"
echo "  - KNN results: ${OUTPUT_PATH}"
echo "=========================================="