import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache
from turbo_cache import TurboKVCache
import time
import pandas as pd
import yaml
import os
import random

def load_config():
    default_config = {
        "model_id": "mlx-community/gemma-2-2b-it-4bit",
        "target_context_length": 16000,
        "num_prompts": 5,
        "max_gen_tokens": 50
    }
    if os.path.exists("config.yml"):
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            return {**default_config, **config}
    return default_config

def get_context_filler(target_len, tokenizer):
    subjects = ["Transformer", "KV cache", "Apple Silicon", "Quantization", "Metal kernel", "MLX", "PolarQuant", "QJL", "Unified Memory", "Attention mechanism", "MatFormer", "NPU", "VRAM", "Throughput", "Latency", "Gradient descent", "Activation function", "Embeddings", "Tokenizer", "Inference"]
    verbs = ["optimizes", "revolutionizes", "scales", "reduces", "implements", "accelerates", "manages", "allocates", "quantizes", "parallelizes", "enhances", "distributes", "computes", "linearizes", "stabilizes"]
    objects = ["memory bandwidth", "inference throughput", "latency", "bit-precision", "multimodal activations", "tensor allocation", "gradient flow", "residual connections", "context window", "on-device performance", "floating point error", "normalization layer", "subspace representation", "elastic inference", "hardware utilization"]
    adjectives = ["highly efficient", "state-of-the-art", "low-latency", "near-lossless", "recursive", "asynchronous", "deterministic", "hardware-accelerated", "nonlinear", "geometric", "probabilistic", "sparse", "dense", "stochastic", "parallel"]
    
    sentences = []
    current_tokens = 0
    while current_tokens < target_len:
        s = f"The {random.choice(adjectives)} {random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} in {random.choice(adjectives)} environments."
        sentences.append(s)
        current_tokens += len(tokenizer.encode(s))
        
    full_text = " ".join(sentences)
    tokens = tokenizer.encode(full_text)
    return tokenizer.decode(tokens[:target_len])

def run_benchmark():
    config = load_config()
    model_id = config["model_id"]
    target_len = config["target_context_length"]
    n_prompts = config["num_prompts"]
    max_tokens = config["max_gen_tokens"]

    print(f"🚀 Loading model: {model_id}...")
    try:
        model, tokenizer = load(model_id)
    except Exception as e:
        print(f"❌ Failed to load model {model_id}: {e}")
        return

    base_prompts = ["Explain memory efficiency.", "Describe LLM scaling.", "What is on-device AI?", "How to optimize kernels?", "Summarize self-attention."]
    while len(base_prompts) < n_prompts:
        base_prompts.extend(base_prompts)
    base_prompts = base_prompts[:n_prompts]

    print(f"📝 Generating high-entropy technical context (~{target_len} tokens)...")
    filler = get_context_filler(target_len, tokenizer)
    
    results = []
    num_layers = len(model.layers)
    head_dim = model.args.head_dim
    
    print(f"📊 Running benchmark for {n_prompts} prompts...")
    
    for i, base_p in enumerate(base_prompts):
        prompt = filler + "\n\nTask: " + base_p
        input_len = len(tokenizer.encode(prompt))
        
        # Clear cache before baseline
        mx.metal.clear_cache()
        
        # --- PASS 1: BASELINE ---
        print(f"[{i+1}/{n_prompts}] Context: {input_len} tokens. Testing Baseline...", end="\r")
        baseline_caches = [KVCache() for _ in range(num_layers)]
        start_t = time.time()
        for _ in generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, prompt_cache=baseline_caches):
            pass
        latency_baseline = time.time() - start_t
        
        # Clear cache before turbo
        mx.metal.clear_cache()
        
        # --- PASS 2: TURBOQUANT ---
        print(f"[{i+1}/{n_prompts}] Context: {input_len} tokens. Testing TurboQuant...", end="\r")
        turbo_caches = [TurboKVCache(d_head=head_dim) for _ in range(num_layers)]
        start_t = time.time()
        for _ in generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, prompt_cache=turbo_caches):
            pass
        latency_turbo = time.time() - start_t
        
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
            "TPS Base": max_tokens / latency_baseline,
            "TPS Turbo": max_tokens / latency_turbo
        })

    print("\n\n✅ Benchmark Complete!")
    df = pd.DataFrame(results)
    df["RAM Reduction"] = df["Baseline RAM (MB)"] / df["Turbo RAM (MB)"]
    
    summary = pd.DataFrame([{
        "Model": model_id,
        "Avg Context": round(df["Input Len"].mean(), 1),
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
    run_benchmark()
