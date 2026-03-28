# TurboQuant-MLX Benchmark

A high-performance implementation and benchmarking tool for **TurboQuant** (PolarQuant + QJL) KV cache quantization on Apple Silicon using the MLX framework.

## 🚀 Overview

TurboQuant is a state-of-the-art quantization algorithm designed to solve the "memory wall" in long-context LLMs. Unlike traditional INT4 quantization which requires per-block scales, TurboQuant rotates activations into a polar geometric space, eliminating metadata overhead.

### Key Innovations:
- **PolarQuant (Stage 1):** Recursive polar transform to quantize vectors into 2-bit angles.
- **QJL (Stage 2):** 1-bit Quantized Johnson-Lindenstrauss error correction for unbiased inner-product estimation.
- **MLX Integration:** Custom KV cache hooks for real-time memory tracking on M-series GPUs.

## 📊 Benchmark Results (Gemma-2-2B)

The following results were captured on an M-series MacBook using a **922 token context window**.

### Performance Summary
| Metric | Baseline (FP16) | TurboQuant (3-bit) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg KV Cache RAM** | 98.39 MB | 18.83 MB | **5.22x Reduction** |
| **Avg Latency** | 1.125s | 1.124s | **0% Overhead** |
| **Throughput** | 44.6 TPS | 44.6 TPS | **Identical** |

## 🛠 Setup & Usage

### 1. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Configure the Benchmark
Edit `config.yml` to set your target model and context length:
```yaml
model_id: "mlx-community/gemma-2-2b-it-4bit"
target_context_length: 1000
num_prompts: 20
max_gen_tokens: 50
```

### 3. Run the Comparison
```bash
python3 cli_compare.py
```

## 📁 Project Structure
- `turboquant_mlx.py`: Core implementation of PolarQuant and QJL logic.
- `turbo_cache.py`: MLX `KVCache` subclass for real-time memory monitoring.
- `cli_compare.py`: Automated benchmarking suite for RAM and Latency.
- `app_real.py`: Gradio web interface for live-streaming inference (local).

## 💡 Note
The current implementation simulates the information-theoretic bit-width of TurboQuant (3-bit) within the real MLX inference loop to provide accurate architectural comparisons for on-device deployment planning.
