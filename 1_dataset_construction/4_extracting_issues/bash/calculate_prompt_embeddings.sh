#!/bin/sh

# Absolute paths
BASE_DIR="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/2_relevance_filtering/data/filter_eval_Kalman"
SCRIPT_PATH="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/4_extracting_issues/src/1_calculate_prompt_embeddings.py"

# Verify the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: Script not found at $SCRIPT_PATH"
  exit 1
fi

CACHE_DIR="/Users/greg/Desktop/newIB/issuebench/cache"
ENV_DIR="/Users/greg/Desktop/newIB/issuebench/env"

# Use virtualenv's python if available, otherwise fallback to system python
PYTHON="$ENV_DIR/bin/python"
if [ -x "$PYTHON" ]; then
  echo "Using Python from virtualenv: $PYTHON"
else
  PYTHON=python
  echo "Virtualenv python not found, using system python"
fi

# Point at the all_clean_filtered.csv file
INPUT_CSV="$BASE_DIR/all_clean_filtered.csv"
OUTPUT_PT="$BASE_DIR/all_clean_filtered_embeddings.pt"

$PYTHON "$SCRIPT_PATH" \
    --input_path "$INPUT_CSV" \
    --num_samples 0 \
    --input_col "user_prompt" \
    --embedding_model all-mpnet-base-v2 \
    --batch_size 32 \
    --cache_dir "$CACHE_DIR" \
    --output_path "$OUTPUT_PT" \
    --seed 123

echo "✅ Embeddings written to $OUTPUT_PT"