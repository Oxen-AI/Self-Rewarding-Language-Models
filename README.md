# 🐂 Oxen.ai Self-Rewarding Language Models 🔁

This is work done by the [Oxen.ai](https://oxen.ai/community) Community, trying to reproduce the [Self-Rewarding Language Model](https://arxiv.org/abs/2401.10020) paper from MetaAI.

<img src="./images/SRLM.png" width="512px"></img>

## 🤖 Goal

The goal is to have a single script that can take in a base LLM and put it into a Self-Reward loop.

## 🏃‍➡️ Steps

0) [00_sft.py](scripts/00_sft.py) - Supervised Fine-Tuning (SFT) of a base model to give it instruction following and evaluation skills.
1) [01_gen_prompts.py](scripts/01_gen_prompts.py) - Generate new prompts to add to the training set.
2) [02_gen_prompts.py](scripts/02_gen_responses.py) - Generate N Responses per prompt, so that we can create preference pairs.
3) [03_gen_scores.py](scripts/03_gen_scores.py) - Score each response from 1-5 for how well it answered the prompt.
4) [04_gen_preferences.py](scripts/04_gen_preferences.py) - Generate preference pairs given the scores to create a DPO dataset
5) [05_dpo.py](scripts/05_dpo.py) - Run Direct Preference Optimization (DPO) to train the next iteration of the model

## 💾 Setup Data Pipeline

Initialize Oxen.ai data repository and the folder for M0.

```bash
mkdir -p data/MO/train
cd data
oxen init
oxen config --set-remote origin https://hub.oxen.ai/$USERNAME/$REPOSITORY
cd ..
```

Download the initial dataset

```
oxen download datasets/Self-Rewarding-Language-Models M0/train/ift+eft.jsonl -o data/M0/train
oxen download datasets/Self-Rewarding-Language-Models M0/train/ift.jsonl -o data/M0/train
```

## ⚽️ Kick it off

Run the self-reward.sh script to generate the first end to end model

```bash
./self-reward.sh scripts mistralai/Mistral-7B-v0.1 M0
```

TODO: Put this in a loop for M0, M1, M2, etc...