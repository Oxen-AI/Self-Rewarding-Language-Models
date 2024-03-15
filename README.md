# Self-Rewarding Language Models

This is work done by the [Oxen.ai](https://oxen.ai/community) Community, trying to reproduce the [Self-Rewarding Language Model](https://arxiv.org/abs/2401.10020) paper from MetaAI.

<img src="./images/diagram.png" width="512px"></img>

## 🤖 Goal 🔁

The goal is to have a single script that can take in a base LLM and put it into a Self-Reward loop.

## Setup Data Pipeline

Initialize Oxen.ai data repository and the folder for M0.

```bash
mkdir -p data/MO
cd data
oxen init
oxen config --set-remote origin https://hub.oxen.ai/$USERNAME/$REPOSITORY
cd ..
```

Download the initial dataset

```
oxen download datasets/Self-Rewarding-Language-Models M0/train/ift+eft.jsonl
```

## Kick it off

Run the self-reward.sh script to generate the first end to end model

```bash
./self-reward.sh scripts/ mistralai/Mistral-7B-v0.1 data/M0
```

TODO: Put this in a loop for M0, M1, M2, etc...