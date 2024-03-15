
CODE=$1
BASE_MODEL=$2
DATA=$3

export PYTHONPATH=./src

# Train the instruction following and evaluation skills
python $CODE/00_sft.py -d $DATA/train/ift+eft.jsonl -m $BASE_MODEL -o $DATA/models/sft

# Upload model to oxen
oxen add $DATA/models/sft
oxen commit -m "adding model $DATA"
oxen push origin main

# Generate new prompts from a base LM and original ift data
# This is hard coded to mistralai/Mistral-7B-v0.1 for now, but could be any base LLM
mkdir $DATA/generated
python $CODE/01_gen_prompts.py mistralai/Mistral-7B-v0.1 $DATA/train/ift.jsonl $DATA/generated/prompts.jsonl

# Upload prompts to oxen
oxen add $DATA/generated/prompts.jsonl
oxen commit -m "adding generated prompts for $DATA"
oxen push origin main

# Generate responses for the prompts
python $CODE/02_gen_responses.py $DATA/models/sft/final_checkpoint $DATA/generated/prompts.jsonl $DATA/generated/responses.jsonl

# Upload responses to oxen
oxen add $DATA/generated/responses.jsonl
oxen commit -m "adding generated responses for $DATA"
oxen push origin main

# Generate responses for the prompts
python $CODE/03_gen_scores.py $DATA/models/sft/final_checkpoint $DATA/generated/responses.jsonl $DATA/generated/scores.jsonl

# Upload scores to oxen
oxen add $DATA/generated/scores.jsonl
oxen commit -m "adding generated scores for $DATA"
oxen push origin main

# Create DPO data
python $CODE/04_gen_preferences.py $DATA/generated/scores.jsonl $DATA/generated/preferences.jsonl

# Upload preferences to oxen
oxen add $DATA/generated/preferences.jsonl
oxen commit -m "adding preferences for $DATA"
oxen push origin main

# Train DPO model
python $CODE/05_dpo.py $DATA/models/sft/final_checkpoint $DATA/generated/preferences.jsonl $DATA/models/dpo/

# Upload DPO Model
oxen add $DATA/models/dpo
oxen commit -m "adding DPO model for $DATA"
oxen push origin main