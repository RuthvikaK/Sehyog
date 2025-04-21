import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
from vllm import LLM, SamplingParams
import time
import sys
import together
import json
from tqdm import tqdm

from qwen import load_qwen_vllm, run_qwen_vllm
from gemma import load_gemma_model, run_gemma
from mistral import load_mistral_model, run_mistral
from llama import load_llama_hf, run_llama_hf

"""
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
"""
#def format_prompt(user_query):
#    return f"""<|system|>
#You are a helpful AI assistant. Provide detailed answers using clear reasoning.
#<|user|>
#{user_query}
#<|assistant|>
#"""
"""
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

def initialize_models():
    models = {}
    print("Warming up models...")

    models["Qwen"] = load_qwen_vllm()
    _ = run_qwen_vllm(models["Qwen"], "Warmup", max_tokens=10)

    models["Gemma"] = load_gemma_model()
    _ = run_gemma(models["Gemma"][0], models["Gemma"][1], "Warmup", ["Option A", "Option B"], max_tokens=10)

    models["Mistral"] = load_mistral_model()
    _ = run_mistral(models["Mistral"][0], models["Mistral"][1], "Warmup", ["Option A", "Option B"], max_tokens=10)
    
    models["LLaMA"] = load_llama_hf()
    _ = run_llama_hf(models["LLaMA"][0], models["LLaMA"][1], "Warmup", max_tokens=10)

    return models
    
"""

def prepare_prompt(query):
    prompt = f"""You are a classifier tasked with a question. Your objective is to read each question carefully and assign it to one of the five predefined categories. The categories reflect different types of reasoning and knowledge assessment styles.

    Your classification should consider the following criteria:
    1. Understanding: Accurately comprehend the core task or question being asked. Is it factual, logical, mathematical, conceptual, or commonsense in nature? 
    2. Alignment: Determine which of the five categories best matches the underlying reasoning or knowledge requirement of the question. 
    3. Specificity: Ensure that the classification is based on the dominant skill tested by the question (e.g., math vs. world knowledge vs. logical reasoning).
    4. Consistency: Apply the same decision-making standards across all questions, even if multiple categories might seem partially applicable.

    You will be given:
    - A question

    Evaluate the following example:
    - Question: {query}

    Instructions:
    - Classify the given question into one of the following categories:
        - Mathematics
        - Science
        - Common Sense
        - Natural Language Understanding
        - Cognitive Reasoning

    Output Format:
    {{
        "category": "category"
    }}
    """
    return prompt

def get_response(api_key, model, prompt, keys, max_attempts=3):
    client = together.Together(api_key=api_key)
    attempts = 0
    response_object = None
    
    system_prompt = "You are an impartial classifier that strictly follows instructions and returns structured JSON as requested."
    
    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON content between brackets
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    response_object = json.loads(json_content)
                    
                    # Check if the response has the expected keys
                    if "category" in response_object:
                        return response_object["category"]
                
                print(f"Attempt {attempts+1}: Could not extract proper JSON format")
                attempts += 1
                
            except json.JSONDecodeError:
                print(f"Attempt {attempts+1}: Invalid JSON format")
                attempts += 1
                
        except Exception as e:
            print(f"Attempt {attempts+1}: Error: {e}")
            attempts += 1
    
    # Return a default category if all attempts fail
    print(f"All attempts failed for query: {prompt[:100]}...")
    return "Unknown"
    
def classifier_of_query(query):
#    print("test")
    api_key = "630e0bb3b0990263b9ab779ca9e80376388b627d571c1c922e72b366dbf918a8"
    model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    keys = ["category"]
    
    prompt = prepare_prompt(query)
#    print(prompt)
    category = get_response(api_key, model, prompt, keys)
#    print(category)
    return category

if __name__ == "__main__":
    print("----------------------------------------------------------------------------------------------------------------------")
    try:
        if len(sys.argv) < 2:
            raise ValueError("No query provided via command line.")
        query = sys.argv[1]
        print("Received CLI input:", query)
      
        category = classifier_of_query(query)
        print(category)
        print("----------------------------------------------------------------------------------------------------------------------")
#        llm = load_qwen_vllm()
#        breakpoint()
#        _ = run_qwen_vllm(llm, "Warmup", max_tokens=10)
        
#        models = initialize_models()
        
        if category == "Mathematics":
            llm = load_qwen_vllm()
            response = run_qwen_vllm(llm, query, max_tokens=128)
            print("qwen")
            del llm
        elif category == "Science":
            tokenizer, model = load_gemma_model()
            response = run_gemma(tokenizer, model, query, ["Option A", "Option B"], max_tokens=128)
            print("gemma")
            del tokenizer, model
        elif category == "Common Sense":
            tokenizer, model = load_mistral_model()
            response = run_mistral(tokenizer, model, query, ["Option A", "Option B"], max_tokens=128)
            print("mistral")
            del tokenizer, model
        elif category == "Natural Language Understanding":
            model, tokenizer = load_llama_hf()
            response = run_llama_hf(model, tokenizer, query, max_tokens=128)
            print("llama")
            del model, tokenizer
        elif category == "Cognitive Reasoning":
            tokenizer, model = load_mistral_model()
            response = run_mistral(tokenizer, model, query, ["Option A", "Option B"], max_tokens=128)
            print("mistral")
            del tokenizer, model
        else:
            llm = load_qwen_vllm()
            response = run_qwen_vllm(llm, query, max_tokens=128)
            print("qwen")
            del llm

        # Free memory
        torch.cuda.empty_cache()
            
        print("Final response:", response)
        with open("output.txt", "w") as f:
            f.write(response)
        
    except Exception as e:
        print(f"Error: {e}")
        if "CUDA out of memory" in str(e):
            print("Try reducing max_tokens or using a smaller model.")