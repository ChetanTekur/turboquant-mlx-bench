import mlx_lm
import mlx.core as mx
import json
from pathlib import Path
from mlx_lm.utils import load_model, get_model_path

def patch_load(model_id):
    """
    Patched load function to handle the Gemma 3N weight structure bug in mlx-lm.
    """
    model_path = get_model_path(model_id)
    
    with open(Path(model_path) / "config.json", "r") as f:
        config = json.load(f)
    
    # Load raw weights
    weights = mx.load(str(Path(model_path) / "model.safetensors"))
    
    # If the sanitize function expects weights["model"] but we have a flat dict
    if "model.embed_tokens.weight" in weights and "model" not in weights:
        print("🔧 Patching weight structure for Gemma 3N...")
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                # Keep keys as is if they already have model. prefix?
                # Actually gemma3n.py sanitize does: weights["model"].pop(k, None)
                # This implies it expects a dict at weights["model"]
                pass
        
        # Let's try to just wrap everything in a "model" key
        weights = {"model": weights}

    # Now manually call the loading logic from mlx_lm.utils
    # This is a bit complex as load_model is internal, but let's try to 
    # just use the standard load with a manual weights override if possible.
    # Alternatively, we can fix the weights on disk or in memory before the call.
    
    return mlx_lm.load(model_id) # This still calls sanitize internally

# Actually, the best way is to monkeypatch the model's sanitize method or the mx.load
