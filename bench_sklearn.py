# bench_numpy.py
import time, argparse, statistics
import numpy as np

def now(): return time.perf_counter()

def matmul_np(N, runs=10, warmups=5):
    A = np.random.rand(N,N).astype(np.float32)
    B = np.random.rand(N,N).astype(np.float32)
    for _ in range(warmups):
        C = A @ B
    times = []
    for _ in range(runs):
        t0 = now()
        C = A @ B
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def add_np(N, runs=10, warmups=5):
    A = np.random.rand(N,N).astype(np.float32)
    B = np.random.rand(N,N).astype(np.float32)
    for _ in range(warmups):
        C = A + B
    times = []
    for _ in range(runs):
        t0 = now()
        C = A + B
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=2048)
    args = parser.parse_args()
    N = args.N
    mat_time = matmul_np(N)
    add_time = add_np(N)
    print(f"NumPy (CPU) N={N}")
    print(f" matmul: time={mat_time:.6f}s, GFLOPS={2*N**3/mat_time/1e9:.2f}")
    print(f" add   : time={add_time:.6f}s, GFLOPS={N**2/add_time/1e9:.2f}")
