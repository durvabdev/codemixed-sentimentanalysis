#!/bin/bash

datasets=("dev" "train" "test")
embeddings=("fasttext" "mbert")
mkdir -p logs

for dataset in "${datasets[@]}"; do
  for embedding in "${embeddings[@]}"; do
    logfile="logs/${dataset}_${embedding}.log"
    echo "Running with dataset: $dataset and embedding: $embedding" > "$logfile"
    python src/run.py -d "$dataset" -e "$embedding" >> "$logfile" 2>&1
  done
done
