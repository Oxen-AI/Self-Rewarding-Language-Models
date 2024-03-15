
import torch
from transformers import TextStreamer

def sample(model, tokenizer, prompt, max_tokens=128):
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt},
        ]

        # Apply the same chat template
        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)

        # Tokenize the data
        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to("cuda")

        stop_token = tokenizer("[/INST]")
        stop_token_id = stop_token.input_ids[0]

        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            num_return_sequences=1,
            eos_token_id=[stop_token_id, tokenizer.eos_token_id],
            max_new_tokens=max_tokens
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        # print(f"A: {answer}")

        # print("\n\n")

        return answer