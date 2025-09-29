DATASET=/path/to/dataset

MODEL=/path/to/base/model
# gpt2-small memory decoder can be downloaded in https://huggingface.co/Clover-Hill/MemoryDecoder-gpt2-small
KNN_PATH=/path/to/memory/decoder

OUTPUT_DIR=tmp/

python -m \
    evaluate_joint \
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