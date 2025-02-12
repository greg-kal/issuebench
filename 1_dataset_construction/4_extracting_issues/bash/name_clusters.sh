#!/bin/sh

REPO=$(git rev-parse --show-toplevel)

source activate $REPO/env/bin/activate

DATASET="all_clean_filtered_clusteroverview"
MODEL="gpt-4o-2024-05-13" #gpt-4o-2024-05-13 #gpt-3.5-turbo-0125

for config in "config_4"; do

    python $REPO/src/clustering/3_name_clusters.py \
        --gen_model $MODEL \
        --input_path $REPO/data/clusters/${config}/${DATASET}.csv \
        --output_path $REPO/data/clusters/${config}/${DATASET}_named3.csv \
        --max_workers 10 \
        --num_samples 20

done