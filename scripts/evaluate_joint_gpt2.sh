DATASET=/fs-computility/plm/shared/jqcao/projects/neuralKNN/dataset/wikitext-gpt2

MODEL=/fs-computility/plm/shared/jqcao/models/gpt2/gpt2-xl
KNN_PATH=/fs-computility/plm/shared/jqcao/projects/MemoryDecoder-tmp/checkpoint/memdec-gpt2-small

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