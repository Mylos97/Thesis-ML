#!/bin/bash

PLANS_PER_QUERY=100
INPUT_FILE="train.txt"
OUTPUT_DIR="initialization"

mkdir -p "$OUTPUT_DIR"

for i in {0..30}; do
    query=$((i + 1))
    start=$(( i * PLANS_PER_QUERY + 1 ))
    end=$(( start + PLANS_PER_QUERY - 1 ))
    sed -n "${start},${end}p" "$INPUT_FILE" > "${OUTPUT_DIR}/${query}.txt"
    echo "Written lines ${start}-${end} -> ${OUTPUT_DIR}/${query}.txt"
done

#echo "Done. ${#queries[@]} files written to '${OUTPUT_DIR}/'"
