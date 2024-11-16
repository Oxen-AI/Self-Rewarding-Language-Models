
from datasets import load_dataset
import time
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import DPOConfig, DPOTrainer

import sys, os

if len(sys.argv) != 5:
    print("Usage: python dpo.py <base_model_name> <model_name> <dataset.json> <results_dir>")
    exit()

base_model_name = sys.argv[1]
model_name = sys.argv[2]
# use $MODEL/generated/preferences.jsonl for the dataset
dataset_file = sys.argv[3]
output_dir = sys.argv[4]

# load the training dataset
dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train'].shuffle(seed=42)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)
base_model.config.use_cache = False

# when running with ref model whis error comes:
# ValueError: You passed both a ref_model and a peft_config.
# For training PEFT adapters with DPO there is no need to pass a reference model. Please pass `ref_model=None` in case you want to train PEFT adapters.
# ref_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map=device_map,
#     trust_remote_code=True,
# )

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.chat_template = """<|begin_of_text|>
{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
"""
tokenizer.eos_token = "<|eot_id|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def get_prompt(example):
    prompt_sample = [
        {"role": "system", "content": "Respond to the following user query in a comprehensive and detailed way. But first write down your internal thoughts. This must include your draft response and its evaluation. After this, write your final response after \"<R>\"."},
        {"role": "user", "content": example['prompt']}
    ]

    prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example['prompt'] = prompt_for_model

    example['chosen'] = example['chosen'] + tokenizer.eos_token
    example['rejected'] = example['rejected'] + tokenizer.eos_token

    # print(example)

    return example

# filter out examples that do not have the required keys or are empty
dataset = dataset.filter(lambda x: 'prompt' in x and 'chosen' in x and 'rejected' in x and x['prompt'] and x['chosen'] and x['rejected'])
dataset = dataset.map(get_prompt)

# from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
lora_dropout=0.05
lora_alpha=16
lora_r=16
learning_rate=5e-4

batch_size = 1

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_config(base_model)

training_args = DPOConfig(
    output_dir=output_dir,
    logging_steps=1,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=50,
    max_length=2048,
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    peft_config=lora_config,
    train_dataset=dataset
)

# Time the training
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)