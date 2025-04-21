import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
import time
import sys


def load_llama_hf(model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    return model, tokenizer


def format_prompt(user_query):
    return f"""<|system|>
You are a helpful AI assistant. Provide detailed answers using clear reasoning.
<|user|>
{user_query}
<|assistant|>
"""


def run_llama_hf(model, tokenizer, user_query, max_tokens=500, temperature=0.7, top_p=0.9, raw_prompt=False):
    if raw_prompt:
        prompt = user_query
    else:
        prompt = format_prompt(user_query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            do_sample=True
        )
    output_time = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    print(f"\nInference time: {output_time:.2f}s ({max_tokens / output_time:.2f} tokens/s)")
    return response


if __name__ == "__main__":
    try:
        model, tokenizer = load_llama_hf()
        _ = run_llama_hf(model, tokenizer, "Warmup", max_tokens=10)
        query = sys.argv[1]
        response = run_llama_hf(model, tokenizer, query, max_tokens=128)
        with open("output.txt", "w") as f:
            f.write(response)
        
    except Exception as e:
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("Try reducing max_tokens or using a smaller model.")
