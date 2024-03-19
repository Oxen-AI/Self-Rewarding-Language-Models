
import argparse
import os
from srlm.trainer import Trainer
from srlm.model import load_model, create_peft_model

from datasets import load_dataset

def collate_fn(tokenizer, x):
    text = tokenizer.apply_chat_template([
        # {"role": "system", "content": "You are a safe, competent, and confident AI."},
        {"role": "user", "content": x['prompt']},
        {"role": "assistant", "content": x['completion']},
    ], tokenize=False)
    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description='SFT train a model.')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='input sft dataset')
    parser.add_argument('-b', '--base_model', default="mistralai/Mistral-7B-v0.1", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-m', '--model', default="mistralai/Mistral-7B-v0.1", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-o', '--output', required=True, type=str, help='output trained model')
    args = parser.parse_args()

    # you can download the dataset file with:
    # `oxen download datasets/Self-Rewarding-Language-Models M0/train/ift.jsonl`
    dataset_file = args.dataset

    # load the training dataset
    dataset = load_dataset("json", data_files={'train': dataset_file})
    dataset = dataset['train'].shuffle(seed=42)

    # load the model
    model, tokenizer = load_model(args.base_model, args.model)
    dataset = dataset.map(lambda x: collate_fn(tokenizer, x))

    print("First example in the dataset")
    print(dataset['text'][0])

    model, lora_config = create_peft_model(model)
    trainer = Trainer(args.output)
    trainer.train(model, tokenizer, lora_config, dataset)

if __name__ == "__main__":
    main()
