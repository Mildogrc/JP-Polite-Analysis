#!/bin/bash

# Usage: ./refine.sh {base|large} {freeze|unfreeze}

MODEL_SIZE=$1
FREEZE_MODE=$2

if [ -z "$MODEL_SIZE" ] || [ -z "$FREEZE_MODE" ]; then
    echo "Usage: ./refine.sh {base|large} {freeze|unfreeze}"
    exit 1
fi

if [ "$FREEZE_MODE" == "freeze" ]; then
    FREEZE_FLAG="true"
elif [ "$FREEZE_MODE" == "unfreeze" ]; then
    FREEZE_FLAG="false"
else
    echo "Invalid freeze mode: $FREEZE_MODE. Use 'freeze' or 'unfreeze'."
    exit 1
fi

# Determine initial model path
# Assuming standard naming from Stage-1
INITIAL_MODEL="models/model_${MODEL_SIZE}_${FREEZE_MODE}.pth"
# If unfreeze, maybe we want to start from the frozen model? 
# Or maybe the user wants to start from the base BERT?
# The prompt says "Load Stage-1 model from: model/".
# I'll assume we start from the frozen checkpoint if available, or just use the flag.
# Actually, let's just pass the path if it exists.

if [ ! -f "$INITIAL_MODEL" ]; then
    echo "Warning: Stage-1 model $INITIAL_MODEL not found. Training will start from scratch/BERT."
fi

echo "Running refinement with Model: $MODEL_SIZE, Freeze Encoder: $FREEZE_FLAG"

python3 scripts/refine_model.py \
    --model_size "$MODEL_SIZE" \
    --freeze_encoder "$FREEZE_FLAG" \
    --initial_model_path "$INITIAL_MODEL" \
    --epochs 3
