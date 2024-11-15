# uses the m1 model to generate 4 completions for each prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import os, sys

if len(sys.argv) < 3:
    print("Usage: python inference.py <model_name> <prompt> [device]")
    exit()

model_name = sys.argv[1]
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
prompt = sys.argv[2]

if len(sys.argv) > 3:
    device = sys.argv[3]
else:
    device = "cuda" # the device to load the model onto


def load_fined_tuned(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, # meta-llama/Llama-3.2-1B-Instruct
        device_map="auto",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        device_map="auto",
    )
#     tokenizer.chat_template = """<|begin_of_text|>
# {%- for message in messages %}
# {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
# {{- message['content'] + '<|eot_id|>' }}
# {%- endfor %}
# {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
# """
    tokenizer.eos_token = "<|eot_id|>"

    return model, tokenizer

def do_sample(model, tokenizer, prompt):
    # print(f"Got prompt: {prompt}")
    with torch.no_grad():

        if model_name == tokenizer_name:
            prompt_sample = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]
        else:
            prompt_sample = [
                {"role": "system", "content": "Respond to the following user query in a comprehensive and detailed way. But first write down your internal thoughts. This must include your draft response and its evaluation. After this, write your final response after \"<R>\"."},
            ]
        prompt_sample.append({"role": "user", "content": prompt})

        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        # print(f"Prompt for model: `{prompt_for_model}`")
        # print("--------------------------------")
        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Self Instruction Creation
        #  For candidate response generation we sample N = 4 candidate responses with temperature T = 0.7, p = 0.9.
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=4096
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = answer[0]
        answer = answer.split("<|eot_id|>")[0]

        return answer

print(f"🧠 Loading the thinking machine...")
model, tokenizer = load_fined_tuned(model_name)
do_sample(model, tokenizer, prompt)

while True:
    prompt = input("> ")
    do_sample(model, tokenizer, prompt)
