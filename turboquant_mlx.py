import mlx.core as mx
import numpy as np
import time

class TurboQuantizer:
    def __init__(self, d_head, bits_angular=2, bits_qjl=1):
        self.d = d_head
        self.bits_angular = bits_angular
        self.bits_qjl = bits_qjl
        
        # Random Preconditioning (Shared Orthogonal Matrix)
        key = mx.random.key(42)
        self.W = mx.random.normal((d_head, d_head), key=key)
        # QR for orthogonality - FORCE TO CPU (not yet supported on GPU)
        self.W, _ = mx.linalg.qr(self.W, stream=mx.cpu)

    def compress(self, k):
        """
        k: (seq_len, d_head)
        """
        # 1. Random Preconditioning
        k_tilde = k @ self.W
        
        # 2. PolarQuant (Coarse)
        radius = mx.linalg.norm(k_tilde, axis=-1, keepdims=True)
        # Avoid division by zero
        safe_radius = mx.where(radius == 0, 1.0, radius)
        v_normalized = k_tilde / safe_radius
        
        # Simple angular quantization simulation
        levels = 2 ** self.bits_angular
        angles_q = mx.round(v_normalized * (levels - 1)) / (levels - 1)
        
        # 3. QJL (1-bit Residual)
        v_coarse = angles_q * radius
        residual = k_tilde - v_coarse
        qjl_bits = mx.sign(residual)
        
        return {
            "radius": radius.astype(mx.float16),
            "angles": angles_q.astype(mx.float16),
            "qjl": qjl_bits.astype(mx.int8)
        }

    def decompress(self, compressed):
        radius = compressed["radius"]
        angles_q = compressed["angles"]
        qjl_bits = compressed["qjl"]
        
        # Coarse reconstruction
        v_coarse = angles_q * radius
        
        # QJL reconstruction (Simplified scale estimation)
        scale = 0.1 * radius 
        v_res = qjl_bits.astype(mx.float16) * scale
        
        v_tilde_recon = v_coarse + v_res
        
        # 4. Inverse Preconditioning
        return v_tilde_recon @ self.W.T

if __name__ == "__main__":
    # Test the MLX implementation
    d = 128
    seq_len = 1024
    quantizer = TurboQuantizer(d)
    
    # Generate test KV
    k_test = mx.random.normal((seq_len, d))
    
    # Warmup
    _ = quantizer.compress(k_test)
    mx.eval(_)

    # Performance Benchmark
    print("Running Metal-accelerated benchmark...")
    start = time.time()
    for _ in range(100):
        compressed = quantizer.compress(k_test)
        mx.eval(compressed)
    end = time.time()
    
    k_recon = quantizer.decompress(compressed)
    error = mx.mean(mx.abs(k_test - k_recon))
    
    print(f"\nMLX Implementation (Metal Accelerated)")
    print(f"--------------------------------------")
    print(f"Sequence Length: {seq_len}")
    print(f"Head Dimension: {d}")
    print(f"Mean Absolute Error: {error.item():.6f}")
    print(f"Avg Compression Time: {(end-start)/100*1000:.2f}ms")
    
    # Memory check (Simulation)
    # FP16: 2 bytes per element
    mem_fp16 = seq_len * d * 2 
    # TurboQuant (3-bit approx): 
    # Radius (16 bits) + Angles (2 bits * d) + QJL (1 bit * d)
    mem_tq_bits = seq_len * (16 + 2*d + 1*d)
    mem_tq = mem_tq_bits / 8
    
    print(f"Estimated Compression Ratio: {mem_fp16/mem_tq:.2f}x")
    print(f"KV Cache Size (32K context, 32 heads, 30 layers):")
    print(f"  FP16: ~7.86 GB")
    print(f"  TurboQuant: ~{7.86 / (mem_fp16/mem_tq):.2f} GB")
