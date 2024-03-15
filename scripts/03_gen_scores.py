# for each completion from the previous step, uses m1 to generate a score

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import re
import os
import sys

# python scripts/gen_scores.py M0/models/sft M0/generated/responses.jsonl M0/generated/scores.jsonl
if len(sys.argv) != 4:
    print("Usage: python 03_gen_scores.py <model_name> <responses_file> <scores_file>")
    exit()

model_name = sys.argv[1]
responses_file = sys.argv[2]
scores_file = sys.argv[3]

device = "cuda" # the device to load the model onto

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return bnb_config

def load_fined_tuned():
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )

    return model, tokenizer

def do_sample(model, tokenizer, prompt):
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": ""},
        ]

        print("-----------------------------------------------------------------------")
        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        # print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to("cuda")

        streamer = TextStreamer(tokenizer)
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            max_new_tokens=100 # since the score is at the beginning
        )

        # print(f"Q: {prompt}:")
        # print("-------------------------")

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        # print(f"A: {answer}")

        # print("\n\n")

        return answer

model, tokenizer = load_fined_tuned()
model.eval()

df = pd.read_json(path_or_buf=responses_file, lines=True)

file = open('llm_as_a_judge_prompt.txt', 'r')
llm_as_a_judge_prompt_template = file.read()
file.close()

pattern = r"[Ss]core: ([0-5])"
results = []
for index, row in df.iterrows():
    prompt_id = row['prompt_id']
    prompt = row['prompt']
    completion = row['completion']

    print("-------------------------")

    llm_as_a_judge_prompt = llm_as_a_judge_prompt_template.format(prompt=prompt,response=completion)
    answer = do_sample(model, tokenizer, llm_as_a_judge_prompt)

    matches = re.findall(pattern, answer)
    generated_score = int(matches[0]) if matches else -1

    # print(f"Answer {answer}")
    print("Found Score: ", generated_score)

    results.append({
        "prompt_id": prompt_id,
        "prompt": prompt,
        "completion": completion,
        "score": generated_score,
        "reasoning": answer
    })

    # save every time
    df_results = pd.DataFrame(results)
    df_results.to_json(scores_file, orient='records', lines=True)


print("Done!")
