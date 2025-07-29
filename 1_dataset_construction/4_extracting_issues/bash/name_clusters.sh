#!/bin/sh

REPO=$(git rev-parse --show-toplevel)


# source activate $REPO/env/bin/activate
PYTHON=python
INPUT="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/2_relevance_filtering/data/filter_eval_Kalman/all_clean_filtered_clusteroverview.csv"
OUTPUT="/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/2_relevance_filtering/data/filter_eval_Kalman/all_clean_filtered_clusteroverview_named.csv"
MODEL="llama3.1:70b"

$PYTHON "/Users/greg/Desktop/newIB/issuebench/1_dataset_construction/4_extracting_issues/src/3_name_clusters.py" \
    --gen_model "$MODEL" \
    --input_path "$INPUT" \
    --output_path "$OUTPUT" \
    --max_workers 10

echo "✅ Clusters named and saved to $OUTPUT"