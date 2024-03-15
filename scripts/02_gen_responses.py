# uses the m1 model to generate 4 completions for each prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import os, sys

if len(sys.argv) != 4:
    print("Usage: python 02_gen_responses.py <model_name> <prompts_file> <responses_file>")
    exit()

model_name = sys.argv[1]
prompts_file = sys.argv[2]
responses_file = sys.argv[3]

device = "cuda" # the device to load the model onto

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return bnb_config

def load_fined_tuned():
    # ift fine-tuned model
    # base_model_name = "raulc0399/mistral-7b-ift-3"

    # the m1 model
    base_model_name = "raulc0399/mistral-7b-m1-v1"

    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        device_map="auto",
    )

    return model, tokenizer

def do_sample(model, tokenizer, prompt):
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": ""},
        ]

        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        # print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer)

        # Self Instruction Creation
        #  For candidate response generation we sample N = 4 candidate responses with temperature T = 0.7, p = 0.9.
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=224
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = answer[0]

        return answer

def extract_completion_only(answer):
    pattern = f"[/INST]"
    parts = answer.split(pattern)
    if len(parts) > 1:
        return parts[-1]
    else:
        return ""

def trim_completion(completion):
    # find the last newline character and remove everything after it
    if "\n" in completion:
        last_newline = completion.rfind("\n")
        completion = completion[:last_newline]
        return completion.strip()
    else:
        return completion

model, tokenizer = load_fined_tuned()
model.eval()

df_prompts = pd.read_json(path_or_buf=prompts_file, lines=True)
# df_prompts = df_prompts.sample(100).reset_index(drop=True)
# shuffle the dataframe
df_prompts = df_prompts.sample(frac=1).reset_index(drop=True)

completions = []
for index, row in df_prompts.iterrows():
    print(f"Processing prompt {index + 1} of {len(df_prompts)}")

    prompt = row['prompt']
    prompt_id = row['prompt_id']

    # sample 4 times as mentioned in the paper
    for completion_sample in range(4):
        print("-----------------------------------------------------------------------")
        print(f"Processing prompt {index + 1}, completion {completion_sample + 1}")

        answer = do_sample(model, tokenizer, prompt)
        completion = extract_completion_only(answer)
        completion = trim_completion(completion)

        completions.append({"prompt_id": prompt_id, "prompt": prompt, "completion": completion})

        print("\n\n")
        print(f"Extracted completion: {completion}")

        df_completions = pd.DataFrame(completions)

        df_completions.to_json(responses_file, orient='records', lines=True)
