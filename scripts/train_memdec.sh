MODEL_FAMILY="gpt2"
KNN_MODEL_SIZE="small"
KNN_DIMENSION=768
MODEL_SIZE="small"
DATASET=/path/to/dataset

ACCELERATE_CONFIG=./accelerate_config/${MODEL_FAMILY}.yaml
MODEL=/path/to/base/model
KNN_DSTORE_PATH=/path/to/knn_dstore/knn_${MODEL_FAMILY}_train_${KNN_DIMENSION}.arrow
OUTPUT_DIR=/path/to/saved/checkpoints

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train_memdec \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name train \
    --knn_save_path ${KNN_DSTORE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate 1e-3 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --seed 42 \
    --checkpointing_steps "epoch" \
    --report_to none