
from transformers import TrainingArguments
from trl import SFTTrainer
import os

class Trainer:
    def __init__(self, output):
        self.output = output

    def train(
            self,
            model,
            tokenizer,
            lora_config,
            dataset
    ):
        # from https://www.datacamp.com/tutorial/mistral-7b-tutorial
        learning_rate=2e-4
        batch_size = 4
        max_seq_length = 1024

        training_args = TrainingArguments(
            output_dir=self.output,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            logging_steps=1,
            num_train_epochs=2,
            save_steps=50
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text"
        )

        trainer.train()

        output_dir = os.path.join(self.output, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)