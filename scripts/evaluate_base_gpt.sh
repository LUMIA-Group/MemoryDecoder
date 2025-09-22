DATASET=/fs-computility/plm/shared/jqcao/projects/MemoryDecoder/dataset/wikitext-gpt2
MODEL=/fs-computility/plm/shared/jqcao/models/gpt2/gpt2-xl
OUTPUT_DIR=tmp/

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python \
    -m \
    train_base \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --eval_subset test \
    --output_dir ${OUTPUT_DIR} \
    --report_to none