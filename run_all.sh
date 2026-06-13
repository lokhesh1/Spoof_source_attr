#!/bin/bash

PTMS=( "wav2vec2_emo" "wav2vec2" )
DATASET="asvspoof2019"

for ptm in "${PTMS[@]}"
do
    echo "Running feature extraction for $ptm"

    python extract_features.py \
        --ptm "$ptm" \
        --dataset "$DATASET"

    echo "Finished $ptm"
done


