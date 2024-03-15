# uses the ift dataset and the model mistralai/Mistral-7B-Instruct-v0.2 to generate prompts
# uses a 8-shot prompt to generate them

import torch
import pandas as pd
from transformers import TextStreamer
import sys
import uuid

from srlm.model import load_model

if len(sys.argv) != 4:
    print("Usage: python 01_gen_prompts.py <model_name> <train.jsonl> <prompts.jsonl>")
    exit()

model_name = sys.argv[1]
ift_dataset_file = sys.argv[2]
generated_prompts_file = sys.argv[3]

device = "cuda" # the device to load the model onto

num_prompts_to_generate=1000

def read_jsonl_file(file_path):
    """Read a JSONL file into a pandas DataFrame."""
    return pd.read_json(file_path, lines=True)

def save_to_jsonl(df, file_path):
    """Save a DataFrame to a JSONL file."""
    df.to_json(file_path, orient='records', lines=True)

def generate_prompt(examples):
    prompt = """
Come up with a series of tasks and questions. Only the task/question,
no further text/explanation, no additional information.
The task or question should be something a person would ask a chatbot.

"""
    for _, item in enumerate(examples):
        prompt += f"<task>{item}</task>\n"

    return prompt

def do_sample(model, tokenizer, examples):
    with torch.no_grad():
        n_shot_prompt = generate_prompt(examples)
        print("<"*80)
        print(f"{n_shot_prompt}")
        print(">"*80)
        model_inputs = tokenizer(n_shot_prompt, return_tensors="pt").to("cuda")

        streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        # print(f"A: {answer}")

        # print("\n\n")

        return answer

def get_random_prompts(df, num_selections=8):
    all_selected_prompts = df.sample(n=num_selections)['prompt'].tolist()

    return all_selected_prompts

def extract_prompts(answer):
    # find all the prompts between <task> </task> brackets
    print("="*80)
    print("Extracting prompts...")
    print(answer)
    print("="*80)

    prompts = []
    while True:
        pattern = f"<task>"
        start = answer.find(pattern)
        if start == -1:
            break
        end = answer.find("</task>")
        if end == -1:
            break
        prompts.append(answer[start + len(pattern):end])
        answer = answer[end + len("</task>"):]

    print("Prompts extracted:")
    print(prompts)
    return prompts

model, tokenizer = load_model(model_name)

ift_df = read_jsonl_file(ift_dataset_file)
uniq_prompts = set([])
new_prompts = []
while True:
    if len(uniq_prompts) >= num_prompts_to_generate:
        break

    task_prompts = get_random_prompts(ift_df)

    answer = do_sample(model, tokenizer, task_prompts)
    prompts = extract_prompts(answer)

    for prompt in prompts:
        if prompt not in uniq_prompts:
            uniq_prompts.add(prompt)
            prompt_id = str(uuid.uuid4())
            new_prompts.append({"prompt_id": prompt_id, "prompt": prompt, "source": "generated"})

    new_prompts_df = pd.DataFrame(new_prompts)
    save_to_jsonl(new_prompts_df, generated_prompts_file)