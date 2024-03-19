#!/bin/bash

CODE=$1
BASE_MODEL=$2
MODEL_NAME=$3

DATA_DIR=data

export PYTHONPATH=src
export TOKENIZER_MODEL=mistralai/Mistral-7B-v0.1

# Train the instruction following and evaluation skills
python $CODE/00_sft.py -d $DATA_DIR/$MODEL_NAME/train/ift_eft.jsonl -b $TOKENIZER_MODEL -m $BASE_MODEL -o $DATA_DIR/$MODEL_NAME/models/sft

# Upload model to oxen
cd $DATA_DIR
oxen add $MODEL_NAME/models/sft/final_checkpoint
oxen commit -m "adding model $MODEL_NAME"
oxen push origin main
cd ..

# Generate new prompts from a base LM and original ift data
# This is hard coded to mistralai/Mistral-7B-v0.1 for now, but could be any base LLM

if [[ ! -e $DATA_DIR/$MODEL_NAME/generated ]]; then
    mkdir $DATA_DIR/$MODEL_NAME/generated
fi
python $CODE/01_gen_prompts.py $TOKENIZER_MODEL $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/train/ift.jsonl $DATA_DIR/$MODEL_NAME/generated/prompts.jsonl

# Upload prompts to oxen
cd $DATA_DIR
oxen add $MODEL_NAME/generated/prompts.jsonl
oxen commit -m "adding generated prompts for $MODEL_NAME"
oxen push origin main
cd ..

# Generate responses for the prompts
python $CODE/02_gen_responses.py $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/prompts.jsonl $DATA_DIR/$MODEL_NAME/generated/responses.jsonl

# Upload responses to oxen
cd $DATA_DIR
oxen add $MODEL_NAME/generated/responses.jsonl
oxen commit -m "adding generated responses for $MODEL_NAME"
oxen push origin main
cd ..

# Generate responses for the prompts
python $CODE/03_gen_scores.py $TOKENIZER_MODEL $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/responses.jsonl $DATA_DIR/$MODEL_NAME/generated/scores.jsonl

# Upload scores to oxen
cd $DATA_DIR
oxen add $MODEL_NAME/generated/scores.jsonl
oxen commit -m "adding generated scores for $MODEL_NAME"
oxen push origin main
cd ..

# Create DPO data
python $CODE/04_gen_preferences.py $DATA_DIR/$MODEL_NAME/generated/scores.jsonl $DATA_DIR/$MODEL_NAME/generated/preferences.jsonl

# Upload preferences to oxen
cd $DATA_DIR
oxen add $MODEL_NAME/generated/preferences.jsonl
oxen commit -m "adding preferences for $MODEL_NAME"
oxen push origin main
cd ..

# Train DPO model
python $CODE/05_dpo.py $TOKENIZER_MODEL $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/preferences.jsonl $DATA_DIR/$MODEL_NAME/models/dpo/

# Upload DPO Model
cd $DATA_DIR
oxen add $MODEL_NAME/models/dpo
oxen commit -m "adding DPO model for $MODEL_NAME"
oxen push origin main
cd ..

# Filter down and concatenate to own training data
oxen df $DATA_DIR/$MODEL_NAME/generated/scores.jsonl --filter "score > 4" -c 'prompt_id,prompt,completion,source' -o $DATA_DIR/$MODEL_NAME/generated/new_ift.jsonl --add-col 'source:generated-ift:str'
oxen df $DATA_DIR/$MODEL_NAME/train/ift_eft.jsonl --vstack $DATA_DIR/$MODEL_NAME/generated/new_ift.jsonl -o $DATA_DIR/$MODEL_NAME/generated/ift_eft.jsonl
oxen df $DATA_DIR/$MODEL_NAME/train/ift_eft.jsonl --filter 'source == ift || source == generated-ift' -o $DATA_DIR/$MODEL_NAME/train/ift.jsonl