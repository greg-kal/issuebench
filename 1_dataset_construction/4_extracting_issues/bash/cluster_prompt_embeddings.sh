#!/bin/sh

REPO=$(git rev-parse --show-toplevel)

source activate $REPO/env/bin/activate

DATASET="all_clean_filtered"

python $REPO/src/clustering/2_cluster_prompt_embeddings.py \
    --prompts_input_path $REPO/data/filtered/$DATASET.csv \
    --prompts_input_col "user_prompt" \
    --embeddings_input_path $REPO/data/clusters/${DATASET}_embeddings.pt \
    --output_path_prompts $REPO/data/clusters/${DATASET}_clusterdetail.csv \
    --output_path_clusters $REPO/data/clusters/${DATASET}_clusteroverview.csv \
    --compute_embeddings_for_visualisation False \
    --dimensionality_reduction_method "umap" \
    --umap_dim 20 \
    --umap_min_dist 0.0 \
    --umap_n_neighbors 15 \
    --umap_metric "cosine" \
    --pca_dim 20 \
    --hdb_min_cluster_size 15 \
    --hdb_min_samples None \
    --hdb_metric "euclidean" \
    --hdb_cluster_selection_method "leaf" \
    --hdb_epsilon 0.0 \
    --top_n_words 20 \
    --top_n_prompts 3 \
    --random_n_prompts 3 \
    --log_level INFO \
    --seed 123
    