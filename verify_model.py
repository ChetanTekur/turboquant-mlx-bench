from mlx_lm import load, generate
import mlx.core as mx

print("🚀 Loading model: mlx-community/gemma-2-2b-it-4bit...")
try:
    model, tokenizer = load("mlx-community/gemma-2-2b-it-4bit")
    print("✅ Model loaded successfully!")

    prompt = "Explain quantization in one short sentence."
    print(f"\nPrompt: {prompt}")
    print("--- Model Output ---")
    
    # verbose=True will print tokens as they are generated with timing stats
    response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=50)
    
    print("\n✅ Inference complete!")
except Exception as e:
    print(f"❌ Error during model verification: {e}")
