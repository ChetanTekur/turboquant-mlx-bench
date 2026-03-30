import mlx_lm
import sys

model_id = "mlx-community/gemma-3n-E2B-it-4bit"
print(f"Checking model: {model_id}")

try:
    # Try loading with explicit verbose output to see where it fails
    res = mlx_lm.load(model_id)
    print(f"Success! Return type: {type(res)}")
    if isinstance(res, tuple):
        print(f"Tuple length: {len(res)}")
        for i, item in enumerate(res):
            print(f"  Item {i}: {type(item)}")
except Exception as e:
    print(f"Caught error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
