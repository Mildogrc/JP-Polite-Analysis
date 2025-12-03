#!/bin/bash

# Usage: ./inference.sh {base|large} {freeze|unfreeze} [num_examples]

MODEL_SIZE=$1
FREEZE_MODE=$2
NUM_EXAMPLES=${3:-10}

if [ -z "$MODEL_SIZE" ] || [ -z "$FREEZE_MODE" ]; then
    echo "Usage: ./inference.sh {base|large} {freeze|unfreeze} [num_examples]"
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

echo "Running inference with Model: $MODEL_SIZE, Freeze Encoder: $FREEZE_FLAG"

python3 scripts/inference.py \
    --model_size "$MODEL_SIZE" \
    --freeze_encoder "$FREEZE_FLAG" \
    --num_examples "$NUM_EXAMPLES"
