#!/bin/bash

DATASET="asvspoof2019"

for PTM in \
    wav2vec2 \
    wav2vec2_emo \
    wavlm \
    whisper \
    xvector
do
    echo "========================================"
    echo "Running PTM: $PTM"
    echo "========================================"

    python main_single.py --ptm "$PTM" --dataset "$DATASET"
done

echo "All experiments completed."