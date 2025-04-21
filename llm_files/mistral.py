import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import time
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_mistral_model(model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model

def format_prompt(question, choices):
    formatted_choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)])
    return (
        f"<s>[INST] Pick the most appropriate answer for the question below.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}\n\n"
        f"Please select the correct answer from the choices above and explain your reasoning concisely so that your response is complete within 80 tokens. [/INST]"
    )

def extract_answer_and_reasoning(response_text):
    answer_match = re.search(r"([A-B])", response_text)
    answer = answer_match.group(1) if answer_match else "Unknown"
    return answer, response_text

def run_mistral(tokenizer, model, question, choices, max_tokens=80):
    prompt = format_prompt(question, choices)
    print("📥 Prompt:\n", prompt)

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        start_time = time.time()
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start_time

    generated_tokens = generated_ids[:, encoded.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    answer, reasoning = extract_answer_and_reasoning(decoded)

    print(f"🧠 Inference time: {elapsed:.2f}s ({max_tokens/elapsed:.2f} tokens/s)")
    return reasoning

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("❌ Usage: python mistral_infer_single_query.py '<question>' '[\"choice1\", \"choice2\"]'")

        question = sys.argv[1]
        choices = eval(sys.argv[2])  # e.g. '["Tighten screws", "Add a shim"]'

        print("🚀 Question received:", question)
        print("🧩 Choices:", choices)

        tokenizer, model = load_mistral_model()
        _ = run_mistral(tokenizer, model, "Warmup", ["Option A", "Option B"], max_tokens=10)

        response = run_mistral(tokenizer, model, question, choices)
        print("📝 Final Response:\n", response)

        with open("output.txt", "w") as f:
            f.write(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        if "CUDA out of memory" in str(e):
            print("⚠️ Try reducing max_tokens or using a smaller model.")
