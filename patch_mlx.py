import mlx_lm.models.gemma3n as gemma3n
import mlx_lm.utils as utils

_original_sanitize = gemma3n.Model.sanitize

def patched_sanitize(self, weights):
    if "model" not in weights:
        # Check if it's a flat dict with model. prefix
        if any(k.startswith("model.") for k in weights.keys()):
            print("🔧 Auto-wrapping weights in 'model' key for Gemma 3N compatibility...")
            weights = {"model": weights}
    return _original_sanitize(self, weights)

# Apply monkeypatch
gemma3n.Model.sanitize = patched_sanitize
print("✅ Gemma 3N loading patch applied.")
