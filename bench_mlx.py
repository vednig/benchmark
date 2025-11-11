import time, statistics, argparse
import numpy as np
import mlx.core as mx  # <-- correct import for arrays and math ops

def now(): return time.perf_counter()

def matmul_mlx(N, runs=10, warmups=5):
    A = mx.array(np.random.randn(N, N).astype(np.float32))
    B = mx.array(np.random.randn(N, N).astype(np.float32))

    # Warm-up
    for _ in range(warmups):
        C = mx.matmul(A, B)
    mx.eval(C)  # Force computation

    times = []
    for _ in range(runs):
        t0 = now()
        C = mx.matmul(A, B)
        mx.eval(C)  # Block until done
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def add_mlx(N, runs=10, warmups=5):
    A = mx.array(np.random.randn(N, N).astype(np.float32))
    B = mx.array(np.random.randn(N, N).astype(np.float32))

    for _ in range(warmups):
        C = A + B
    mx.eval(C)

    times = []
    for _ in range(runs):
        t0 = now()
        C = A + B
        mx.eval(C)
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=2048)
    args = parser.parse_args()
    N = args.N

    mat_time = matmul_mlx(N)
    add_time = add_mlx(N)

    flops_mat = 2 * (N ** 3)
    flops_add = 1 * (N ** 2)

    print(f"MLX (Apple Silicon, Metal backend), N={N}")
    print(f" matmul: {mat_time:.6f}s → {flops_mat / mat_time / 1e9:.2f} GFLOPS")
    print(f" add   : {add_time:.6f}s → {flops_add / add_time / 1e9:.2f} GFLOPS")
