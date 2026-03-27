#!/bin/bash

queries=(
    "1.sql" "10.sql" "11.sql" "12.sql" "13.sql" "14.sql" "15.sql"
    "16.sql" "17.sql" "18.sql" "19.sql" "2.sql" "20.sql" "21.sql"
    "22.sql" "23.sql" "24.sql" "25.sql" "26.sql" "27.sql" "28.sql"
    "29.sql" "3.sql" "30.sql" "4.sql" "5.sql" "6.sql" "7.sql"
    "8.sql" "9.sql"
)

PLANS_PER_QUERY=100
INPUT_FILE="train.txt"
OUTPUT_DIR="initialization"

mkdir -p "$OUTPUT_DIR"

for i in "${!queries[@]}"; do
    query="${queries[$i]}"
    start=$(( i * PLANS_PER_QUERY + 1 ))
    end=$(( start + PLANS_PER_QUERY - 1 ))
    sed -n "${start},${end}p" "$INPUT_FILE" > "${OUTPUT_DIR}/${query}.txt"
    echo "Written lines ${start}-${end} -> ${OUTPUT_DIR}/${query}.txt"
done

echo "Done. ${#queries[@]} files written to '${OUTPUT_DIR}/'"
