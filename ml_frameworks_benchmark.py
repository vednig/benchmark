import time, statistics, numpy as np

results = []

def now(): return time.perf_counter()

def add_result(framework, device, kernel, N, t, flops):
    gflops = flops / t / 1e9
    results.append({
        "Framework": framework,
        "Device": device,
        "Kernel": kernel,
        "N": N,
        "Time (s)": round(t, 6),
        "GFLOPS": round(gflops, 2)
    })

# ================
# Framework Benchmarks
# ================

def bench_numpy(N, runs=10, warmups=5):
    import numpy as np
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    # matmul
    for _ in range(warmups): C = A @ B
    times = []
    for _ in range(runs):
        t0 = now(); C = A @ B; t1 = now()
        times.append(t1 - t0)
    add_result("NumPy", "CPU", "matmul", N, statistics.median(times), 2 * N**3)
    # add
    for _ in range(warmups): C = A + B
    times = []
    for _ in range(runs):
        t0 = now(); C = A + B; t1 = now()
        times.append(t1 - t0)
    add_result("NumPy", "CPU", "add", N, statistics.median(times), N**2)

def bench_torch(N, runs=10, warmups=5):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn((N, N), dtype=torch.float32, device=device)
    B = torch.randn((N, N), dtype=torch.float32, device=device)
    # matmul
    for _ in range(warmups): C = A @ B
    if device.type == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = now(); C = A @ B
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = now(); times.append(t1 - t0)
    add_result("PyTorch", str(device), "matmul", N, statistics.median(times), 2 * N**3)
    # add
    for _ in range(warmups): C = A + B
    if device.type == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = now(); C = A + B
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = now(); times.append(t1 - t0)
    add_result("PyTorch", str(device), "add", N, statistics.median(times), N**2)

def bench_tf(N, runs=10, warmups=5):
    import tensorflow as tf
    def sync(): _ = tf.constant(0.0).numpy()
    A = tf.random.uniform((N, N), dtype=tf.float32)
    B = tf.random.uniform((N, N), dtype=tf.float32)
    # matmul
    for _ in range(warmups): C = tf.linalg.matmul(A, B)
    sync()
    times = []
    for _ in range(runs):
        t0 = now(); C = tf.linalg.matmul(A, B); sync(); t1 = now()
        times.append(t1 - t0)
    add_result("TensorFlow", "default", "matmul", N, statistics.median(times), 2 * N**3)
    # add
    for _ in range(warmups): C = A + B
    sync()
    times = []
    for _ in range(runs):
        t0 = now(); C = A + B; sync(); t1 = now()
        times.append(t1 - t0)
    add_result("TensorFlow", "default", "add", N, statistics.median(times), N**2)

def bench_mlx(N, runs=10, warmups=5):
    import mlx.core as mx
    A = mx.array(np.random.randn(N, N).astype(np.float32))
    B = mx.array(np.random.randn(N, N).astype(np.float32))
    # matmul
    for _ in range(warmups): C = mx.matmul(A, B)
    mx.eval(C)
    times = []
    for _ in range(runs):
        t0 = now(); C = mx.matmul(A, B); mx.eval(C); t1 = now()
        times.append(t1 - t0)
    add_result("MLX", "Metal", "matmul", N, statistics.median(times), 2 * N**3)
    # add
    for _ in range(warmups): C = A + B
    mx.eval(C)
    times = []
    for _ in range(runs):
        t0 = now(); C = A + B; mx.eval(C); t1 = now()
        times.append(t1 - t0)
    add_result("MLX", "Metal", "add", N, statistics.median(times), N**2)

# ================
# Main Runner
# ================
if __name__ == "__main__":
    N = 2048
    print(f"\n=== Benchmarking ML Frameworks (N={N}) ===\n")

    modules = {
        "NumPy": "numpy",
        "PyTorch": "torch",
        "TensorFlow": "tensorflow",
        "MLX": "mlx.core"
    }

    benches = {
        "NumPy": bench_numpy,
        "PyTorch": bench_torch,
        "TensorFlow": bench_tf,
        "MLX": bench_mlx
    }

    for name, mod in modules.items():
        try:
            __import__(mod)
            print(f"→ Running {name}...")
            benches[name](N)
        except ImportError:
            print(f"× {name} not installed, skipping.")
        except Exception as e:
            print(f"! {name} failed: {e}")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n=== Results (median of runs) ===")
        print(df.to_string(index=False))
    else:
        print("No frameworks ran successfully.")
