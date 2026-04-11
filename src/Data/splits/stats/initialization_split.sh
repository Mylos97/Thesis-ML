#!/bin/bash

BENCHMARK_DIR="/home/juri/itu/thesis-wayang/wayang-plugins/wayang-ml/src/main/resources/benchmarks/stats"
mapfile -t queries < <(ls "$BENCHMARK_DIR"/*.sql | xargs -n1 basename | sort)

PLANS_PER_QUERY=100
INPUT_FILE="train.txt"
OUTPUT_DIR="initialization"

mkdir -p "$OUTPUT_DIR"

excluded_queries=("q595820.sql")

filtered_queries=()
for query in "${queries[@]}"; do
    skip=0
    for excluded in "${excluded_queries[@]}"; do
        if [[ "$query" == "$excluded" ]]; then
            skip=1
            break
        fi
    done
    [[ $skip -eq 0 ]] && filtered_queries+=("$query")
done

for i in "${!filtered_queries[@]}"; do
    query="${filtered_queries[$i]}"
    start=$(( i * PLANS_PER_QUERY + 1 ))
    end=$(( start + PLANS_PER_QUERY - 1 ))
    sed -n "${start},${end}p" "$INPUT_FILE" > "${OUTPUT_DIR}/${query}.txt"
    echo "Written lines ${start}-${end} -> ${OUTPUT_DIR}/${query}.txt"
done

echo "Done. ${#filtered_queries[@]} files written to '${OUTPUT_DIR}/'"

