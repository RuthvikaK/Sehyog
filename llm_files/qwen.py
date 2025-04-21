import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
from vllm import LLM, SamplingParams
import time
import sys

def load_qwen_vllm(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):
    # breakpoint()
    llm = LLM(model_name, tensor_parallel_size=1, dtype="float16")
    return llm

def format_prompt(user_query):
    return (
        "<|system|>\n"
        "You are a helpful AI assistant. Provide detailed answers using clear reasoning.\n"
        "<|user|>\n" +
        user_query +
        "\n<|assistant|>\n"
    )

#def format_prompt(user_query):
#    return f"""<|system|>
#You are a helpful AI assistant. Provide detailed answers using clear reasoning.
#<|user|>
#{user_query}
#<|assistant|>
#"""

def run_qwen_vllm(llm, user_query, max_tokens=500, temperature=0.7, top_p=0.9, raw_prompt=False):
    if raw_prompt:
        prompt = user_query
    else:
        prompt = format_prompt(user_query)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.1
    )
    
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    output_time = time.time() - start_time
    
    response = outputs[0].outputs[0].text.strip()
    print(f"\nInference time: {output_time:.2f}s ({max_tokens/output_time:.2f} tokens/s)")
    return response

if __name__ == "__main__":
    try:
        llm = load_qwen_vllm()
#        breakpoint()
        _ = run_qwen_vllm(llm, "Warmup", max_tokens=10)
        query = sys.argv[1]
        print("Received CLI input:", query)
        response = run_qwen_vllm(llm, query, max_tokens=128)
        print("Final response:", response)
        with open("output.txt", "w") as f:
            f.write(response)
        
    except Exception as e:
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("Try reducing max_tokens or using a smaller model.")
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import time
"""
#def format_prompt(user_query):
#    return f"""<|system|>
#You are a helpful AI assistant. Provide detailed answers using clear reasoning.
#<|user|>
#{user_query}
#<|assistant|>
#"""
"""
def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def run_qwen(tokenizer, model, query, max_tokens=512):
    prompt = format_prompt(query)
    print("📥 Prompt:\n", prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🧠 Inference time: {elapsed:.2f}s")

    # Strip everything before <|assistant|>
    assistant_idx = response.find("<|assistant|>")
    if assistant_idx != -1:
        response = response[assistant_idx + len("<|assistant|>"):].strip()

    return response

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("❌ No query provided!")

        query = sys.argv[1]
        print("🚀 Query received:", query)

        tokenizer, model = load_qwen_model()
        response = run_qwen(tokenizer, model, query)

        print("📝 Final Response:\n", response)

        with open("output.txt", "w") as f:
            f.write(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        if "CUDA out of memory" in str(e):
            print("⚠️ Try reducing max_tokens or using a smaller model.")
"""
