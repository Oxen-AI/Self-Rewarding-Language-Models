
import json
import os
import sys
import uuid

if len(sys.argv) != 3:
    print("Usage: python 04_gen_preferences.py <scores.jsonl> <preferences.jsonl>")
    exit()

scores_file = sys.argv[1]
preferences_file = sys.argv[2]

# Group all the prompts by prompt_id
prompts = {}
with open(scores_file, "r") as f:
    for line in f:
        row = json.loads(line)

        prompt_id = row['prompt_id']
        if prompt_id not in prompts:
            prompts[prompt_id] = []

        prompts[row['prompt_id']].append(row)

# Iterate over prompts and look at high and low scores to generate preference pairs
# if the score is the same, skip
pairs = []
for prompt_id, prompts in prompts.items():
    # find the best score
    best_score = -1
    best_prompt = None
    for prompt in prompts:
        if prompt['score'] > best_score:
            best_score = prompt['score']
            best_prompt = prompt
    # find the worst score
    worst_score = 100
    worst_prompt = None
    for prompt in prompts:
        if prompt['score'] < worst_score:
            worst_score = prompt['score']
            worst_prompt = prompt

    if None == best_prompt or None == worst_prompt:
        continue

    if best_score == worst_score:
        continue

    pairs.append({
        "prompt_id": best_prompt['prompt_id'],
        "prompt": best_prompt['prompt'],
        "chosen": best_prompt['completion'],
        "rejected": worst_prompt['completion'],
        "score_chosen": best_prompt['score'],
        "score_rejected": worst_prompt['score']
    })



with open(preferences_file, "w") as f:
    for line in pairs:
        f.write(json.dumps(line) + "\n")

