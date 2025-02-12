#!/bin/sh

REPO=$(git rev-parse --show-toplevel)

source activate $REPO/env/bin/activate

for DATASET_NAME in "all_clean_filtered"; do

    python $REPO/src/clustering/1_calculate_prompt_embeddings.py \
        --input_path $REPO/data/filtered/$DATASET_NAME.csv \
        --num_samples 0 \
        --input_col "user_prompt" \
        --embedding_model all-mpnet-base-v2 \
        --batch_size 32 \
        --output_path $REPO/data/clusters/${DATASET_NAME}_embeddings.pt \
        --cache_dir $REPO/cache/ \
        --seed 123

done