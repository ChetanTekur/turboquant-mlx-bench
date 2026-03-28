import mlx.core as mx
import numpy as np
import gradio as gr
import time
import pandas as pd
from PIL import Image
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache
from turbo_cache import TurboKVCache

# Load real model and tokenizer
print("Loading model mlx-community/gemma-2-2b-it-4bit...")
model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")

def run_turboquant_demo(image, prompt, max_tokens=256):
    """
    Runs REAL inference using custom TurboKVCache and tracks memory.
    """
    num_layers = len(model.layers)
    turbo_caches = [TurboKVCache(d_head=model.args.head_dim) for _ in range(num_layers)]
    
    # 1. Simulate Vision Encoding (MatFormer 3N)
    if image is not None:
        print("Image detected: Prefilling 1024 vision tokens into KV Cache...")
        # (B, H, L, D)
        dummy_keys = mx.random.normal((1, model.args.num_key_value_heads, 1024, model.args.head_dim))
        dummy_vals = mx.random.normal((1, model.args.num_key_value_heads, 1024, model.args.head_dim))
        for c in turbo_caches:
            c.update_and_fetch(dummy_keys, dummy_vals)

    token_counts = []
    baseline_mb = []
    turbo_mb = []
    response_text = ""
    
    # 2. Run REAL inference
    for token in generate(model, tokenizer, prompt, max_tokens=max_tokens, prompt_cache=turbo_caches):
        response_text += token
        
        total_baseline = 0
        total_turbo = 0
        num_tokens = 0
        
        for c in turbo_caches:
            nt, b, t = c.get_memory_stats()
            num_tokens = nt
            total_baseline += b
            total_turbo += t
            
        token_counts.append(num_tokens)
        baseline_mb.append(total_baseline / (1024 * 1024))
        turbo_mb.append(total_turbo / (1024 * 1024))
        
        # Prepare DataFrame for raw data display (bypassing LinePlot bug)
        df_list = []
        for i in range(len(token_counts)):
            df_list.append({
                "Tokens": token_counts[i], 
                "Baseline (MB)": round(baseline_mb[i], 2), 
                "TurboQuant (MB)": round(turbo_mb[i], 2),
                "Savings (x)": round(baseline_mb[i] / turbo_mb[i], 2) if turbo_mb[i] > 0 else 0
            })
        
        df = pd.DataFrame(df_list)
        # Show only the last 10 rows to keep it readable
        yield response_text, df.tail(10)

def build_app():
    with gr.Blocks(title="TurboQuant REAL DEMO") as demo:
        gr.Markdown("# 🚀 Real-Time TurboQuant Implementation")
        gr.Markdown("Comparing Baseline FP16 vs TurboQuant KV Cache (Real Inference on Gemma-2-2B)")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload Image (Gemma 3N VLM Simulation)")
                input_prompt = gr.Textbox(
                    label="Prompt", 
                    value="Write a short poem about memory efficiency.",
                    lines=3
                )
                run_btn = gr.Button("Run Real Inference", variant="primary")
                
            with gr.Column(scale=2):
                output_text = gr.Textbox(label="Model Output", lines=8)
                gr.Markdown("### 📈 Live KV Cache Stats")
                memory_table = gr.Dataframe(
                    label="Memory Usage (MB)",
                    headers=["Tokens", "Baseline (MB)", "TurboQuant (MB)", "Savings (x)"],
                    interactive=False
                )

        run_btn.click(
            fn=run_turboquant_demo,
            inputs=[input_img, input_prompt],
            outputs=[output_text, memory_table]
        )
        
    return demo

if __name__ == "__main__":
    demo = build_app()
    print("Real-time App initialized with weights. Starting server on localhost:7860...")
    # share=True is banned by Santa. server_name=127.0.0.1 is standard.
    demo.launch(server_name="127.0.0.1", server_port=7860)
