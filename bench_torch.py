# bench_torch.py
import time, argparse, statistics, torch, numpy as np

def now(): return time.perf_counter()

def matmul_torch(N, device, runs=10, warmups=5):
    A = torch.randn((N,N), dtype=torch.float32, device=device)
    B = torch.randn((N,N), dtype=torch.float32, device=device)
    # warmup
    for _ in range(warmups):
        C = A @ B
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = now()
        C = A @ B
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def add_torch(N, device, runs=10, warmups=5):
    A = torch.randn((N,N), dtype=torch.float32, device=device)
    B = torch.randn((N,N), dtype=torch.float32, device=device)
    for _ in range(warmups):
        C = A + B
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = now()
        C = A + B
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def run_once(N, device_str):
    device = torch.device(device_str)
    mat_time = matmul_torch(N, device)
    add_time = add_torch(N, device)
    flops_mat = 2 * (N**3)
    flops_add = 1 * (N**2)
    print(f"PyTorch device={device}, N={N}")
    print(f" matmul: time={mat_time:.6f}s, GFLOPS={flops_mat/mat_time/1e9:.2f}")
    print(f" add   : time={add_time:.6f}s, GFLOPS={flops_add/add_time/1e9:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cpu')  # or 'cuda'
    args = parser.parse_args()
    run_once(args.N, args.device)
