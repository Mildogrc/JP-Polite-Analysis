#!/bin/bash

# Usage: ./train.sh {base|large} {freeze|unfreeze}

MODEL_SIZE=$1
FREEZE_MODE=$2

if [ -z "$MODEL_SIZE" ] || [ -z "$FREEZE_MODE" ]; then
    echo "Usage: ./train.sh {base|large} {freeze|unfreeze}"
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

echo "Running training with Model: $MODEL_SIZE, Freeze Encoder: $FREEZE_FLAG"

python3 scripts/train.py \
    --model_size "$MODEL_SIZE" \
    --freeze_encoder "$FREEZE_FLAG" \
    --epochs 3 \
    --batch_size 32
