# #!/bin/sh
#
# REPO=$(git rev-parse --show-toplevel)
#
# source activate $REPO/env/bin/activate
#
# DATASET="all_clean_filtered"
#
# python $REPO/src/clustering/2_cluster_prompt_embeddings.py \
#     --prompts_input_path $REPO/data/filtered/$DATASET.csv \
#     --prompts_input_col "user_prompt" \
#     --embeddings_input_path $REPO/data/clusters/${DATASET}_embeddings.pt \
#     --output_path_prompts $REPO/data/clusters/${DATASET}_clusterdetail.csv \
#     --output_path_clusters $REPO/data/clusters/${DATASET}_clusteroverview.csv \
#     --compute_embeddings_for_visualisation False \
#     --dimensionality_reduction_method "umap" \
#     --umap_dim 20 \
#     --umap_min_dist 0.0 \
#     --umap_n_neighbors 15 \
#     --umap_metric "cosine" \
#     --pca_dim 20 \
#     --hdb_min_cluster_size 15 \
#     --hdb_min_samples None \
#     --hdb_metric "euclidean" \
#     --hdb_cluster_selection_method "leaf" \
#     --hdb_epsilon 0.0 \
#     --top_n_words 20 \
#     --top_n_prompts 3 \
#     --random_n_prompts 3 \
#     --log_level INFO \
#     --seed 123

#!/bin/sh

# Absolute paths
BASE_DIR="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/2_relevance_filtering/data/filter_eval_Kalman"
SCRIPT_PATH="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/4_extracting_issues/src/2_cluster_prompt_embeddings.py"
ENV_DIR="/Users/greg/Desktop/newIB/issuebench/env"

# Verify the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: Script not found at $SCRIPT_PATH"
  exit 1
fi

echo "Running clustering script at: $SCRIPT_PATH"

# Ensure virtualenv's python is available
PYTHON="$ENV_DIR/bin/python"
if [ -x "$PYTHON" ]; then
  echo "Using Python from virtualenv: $PYTHON"
else
  echo "Error: Virtualenv python not found at $PYTHON"
  echo "Please activate your project's environment or install dependencies (e.g., umap-learn)."
  exit 1
fi

# Dataset settings
DATASET="all_clean_filtered"
PROMPTS_INPUT="$BASE_DIR/${DATASET}.csv"
EMBEDDINGS_INPUT="$BASE_DIR/${DATASET}_embeddings.pt"
OUTPUT_PROMPTS="$BASE_DIR/${DATASET}_clusterdetail.csv"
OUTPUT_CLUSTERS="$BASE_DIR/${DATASET}_clusteroverview.csv"

# Execute clustering
$PYTHON "$SCRIPT_PATH" \
    --prompts_input_path "$PROMPTS_INPUT" \
    --prompts_input_col "user_prompt" \
    --embeddings_input_path "$EMBEDDINGS_INPUT" \
    --output_path_prompts "$OUTPUT_PROMPTS" \
    --output_path_clusters "$OUTPUT_CLUSTERS" \
    --compute_embeddings_for_visualisation False \
    --dimensionality_reduction_method "umap" \
    --umap_dim 20 \
    --umap_min_dist 0.0 \
    --umap_n_neighbors 15 \
    --umap_metric "cosine" \
    --pca_dim 20 \
    --hdb_min_cluster_size 3 \
    --hdb_min_samples None \
    --hdb_metric "euclidean" \
    --hdb_cluster_selection_method "leaf" \
    --hdb_epsilon 0.0 \
    --top_n_words 20 \
    --top_n_prompts 3 \
    --random_n_prompts 3 \
    --log_level INFO \
    --seed 123

echo "✅ Clustering complete: details at $OUTPUT_PROMPTS and overview at $OUTPUT_CLUSTERS"