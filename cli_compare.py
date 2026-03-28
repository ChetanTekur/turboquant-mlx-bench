import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache
from turbo_cache import TurboKVCache
import time
import pandas as pd
import random

# A long technical filler to pad context to ~1000 tokens
LONG_FILLER = """
The Transformer architecture, introduced in the seminal paper 'Attention is All You Need', 
revolutionized the field of natural language processing by replacing recurrent neural networks 
and convolutional layers with a mechanism known as self-attention. This mechanism allows the model 
to weight the importance of different words in a sequence relative to each other, regardless of 
their distance. The core components of a Transformer block include multi-head self-attention, 
layer normalization, and feed-forward neural networks. Each attention head performs a scaled 
dot-product operation: Attention(Q, K, V) = softmax((QK^T)/sqrt(dk))V. This allows the model to 
parallelize computations effectively, leading to significant speedups during training and inference. 
Large Language Models (LLMs) like GPT-4, Llama, and Gemma leverage these Transformer blocks to 
scale to billions of parameters, capturing complex linguistic patterns and world knowledge. 
As context windows grow to 32K, 128K, or even 1 million tokens, the memory footprint of the 
Key-Value (KV) cache becomes the primary bottleneck for on-device deployment. The KV cache 
stores the keys and values of previous tokens to avoid redundant computations during auto-regressive 
generation. In FP16, this cache grows linearly with context length, often exceeding the VRAM 
capacity of mobile devices. TurboQuant addresses this by applying PolarQuant and Quantized 
Johnson-Lindenstrauss error correction, achieving up to 6x reduction in KV cache size while 
maintaining near-lossless accuracy. By rotating activations into a polar geometric space, 
it eliminates the metadata overhead typically associated with traditional quantization. 
This technology enables high-performance, long-context AI on consumer-grade hardware.
""" * 5 

def run_benchmark(n_prompts=20):
    print(f"🚀 Loading model: mlx-community/gemma-2-2b-it-4bit...")
    model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")
    
    base_prompts = [
        "Explain the importance of low-latency AI.",
        "How does quantization work in neural networks?",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of Apple Silicon for ML?",
        "Summarize the history of deep learning.",
        "How do transformers handle long sequences?",
        "Explain backpropagation like I am five.",
        "What is the difference between FP16 and INT4?",
        "Write a Python function to sort a list.",
        "Describe the architecture of Gemma 2B.",
        "What is a KV cache in LLMs?",
        "How does flash attention speed up inference?",
        "Explain the concept of weight tying.",
        "What are residual connections?",
        "Write a haiku about memory bandwidth.",
        "Describe the impact of MoE models.",
        "What is the goal of distal supervision?",
        "How does LoRA fine-tuning work?",
        "Explain the role of the tokenizer.",
        "What is temperature in LLM sampling?"
    ]
    
    results = []
    num_layers = len(model.layers)
    head_dim = model.args.head_dim
    
    print(f"📊 Running benchmark for {n_prompts} prompts...")
    print(f"🎯 Target Context Length: >1000 tokens\n")
    
    for i, base_p in enumerate(base_prompts[:n_prompts]):
        prompt = LONG_FILLER + "\n\nTask: " + base_p
        input_len = len(tokenizer.encode(prompt))
        
        # --- PASS 1: BASELINE ---
        print(f"[{i+1}/{n_prompts}] Input: {input_len} tokens. Testing Baseline...", end="\r")
        baseline_caches = [KVCache() for _ in range(num_layers)]
        start_t = time.time()
        out_len = 0
        for _ in generate(model, tokenizer, prompt=prompt, max_tokens=50, prompt_cache=baseline_caches):
            out_len += 1
        latency_baseline = time.time() - start_t
        
        # --- PASS 2: TURBOQUANT ---
        print(f"[{i+1}/{n_prompts}] Input: {input_len} tokens. Testing TurboQuant...", end="\r")
        turbo_caches = [TurboKVCache(d_head=head_dim) for _ in range(num_layers)]
        start_t = time.time()
        for _ in generate(model, tokenizer, prompt=prompt, max_tokens=50, prompt_cache=turbo_caches):
            pass
        latency_turbo = time.time() - start_t
        
        # Memory Stats from TurboPass (which tracks both)
        total_baseline_mb = 0
        total_turbo_mb = 0
        for cache in turbo_caches:
            _, b, t = cache.get_memory_stats()
            total_baseline_mb += b / (1024*1024)
            total_turbo_mb += t / (1024*1024)
            
        results.append({
            "Input Len": input_len,
            "Baseline RAM (MB)": total_baseline_mb,
            "Turbo RAM (MB)": total_turbo_mb,
            "Latency Base (s)": latency_baseline,
            "Latency Turbo (s)": latency_turbo,
            "TPS Base": 50 / latency_baseline,
            "TPS Turbo": 50 / latency_turbo
        })

    print("\n\n✅ Benchmark Complete!")
    df = pd.DataFrame(results)
    df["RAM Reduction"] = df["Baseline RAM (MB)"] / df["Turbo RAM (MB)"]
    
    # Calculate Averages
    summary = pd.DataFrame([{
        "Avg Input": round(df["Input Len"].mean(), 1),
        "Avg Baseline RAM": f"{df['Baseline RAM (MB)'].mean():.2f} MB",
        "Avg Turbo RAM": f"{df['Turbo RAM (MB)'].mean():.2f} MB",
        "Avg RAM Reduction": f"{df['RAM Reduction'].mean():.2f}x",
        "Avg Latency Base": f"{df['Latency Base (s)'].mean():.3f}s",
        "Avg Latency Turbo": f"{df['Latency Turbo (s)'].mean():.3f}s",
        "Avg TPS Base": round(df["TPS Base"].mean(), 1),
        "Avg TPS Turbo": round(df["TPS Turbo"].mean(), 1)
    }])
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(summary.to_string(index=False))
    
    print("\n--- DETAILED RESULTS ---")
    print(df[["Input Len", "Baseline RAM (MB)", "Turbo RAM (MB)", "Latency Base (s)", "Latency Turbo (s)"]].round(3).to_string())

if __name__ == "__main__":
    run_benchmark(n_prompts=20)
