import time, statistics, argparse
import tensorflow as tf

def now(): return time.perf_counter()

def sync():
    # Force any pending ops to finish
    # Convert small tensor to numpy (forces synchronization)
    _ = tf.constant(0.0).numpy()

def matmul_tf(N, runs=10, warmups=5):
    A = tf.random.uniform((N, N), dtype=tf.float32)
    B = tf.random.uniform((N, N), dtype=tf.float32)

    # warm-up
    for _ in range(warmups):
        C = tf.linalg.matmul(A, B)
    sync()

    times = []
    for _ in range(runs):
        t0 = now()
        C = tf.linalg.matmul(A, B)
        sync()
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def add_tf(N, runs=10, warmups=5):
    A = tf.random.uniform((N, N), dtype=tf.float32)
    B = tf.random.uniform((N, N), dtype=tf.float32)

    for _ in range(warmups):
        C = A + B
    sync()

    times = []
    for _ in range(runs):
        t0 = now()
        C = A + B
        sync()
        t1 = now()
        times.append(t1 - t0)
    return statistics.median(times)

def run_once(N):
    mat_time = matmul_tf(N)
    add_time = add_tf(N)
    flops_mat = 2 * (N ** 3)
    flops_add = 1 * (N ** 2)

    print(f"TensorFlow default device = {tf.config.list_physical_devices()}")
    print(f"N = {N}")
    print(f" matmul: {mat_time:.6f}s → {flops_mat / mat_time / 1e9:.2f} GFLOPS")
    print(f" add   : {add_time:.6f}s → {flops_add / add_time / 1e9:.2f} GFLOPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2048)
    args = parser.parse_args()
    run_once(args.N)
