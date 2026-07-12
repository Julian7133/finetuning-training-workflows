"""Isolate memory contributors of one training step at T=8192.

Micro-benchmarks with real tensor shapes, no model load. Prints MLX peak
active memory per experiment. Run each experiment in a SUBPROCESS (fresh
process -> clean peak counter and allocator) via: python diag_mem.py <name>

History (peaks at T=8192, one layer / one loss):
  sdpa fwd+bwd                       6.81 GB   (T x T materialized in bwd)
  mx.checkpoint-chunked delta       17.90 GB   (scheduler runs recomputes
  mx.checkpoint-chunked delta (mx.compile)  22.38 GB    concurrently!)
  mx.checkpoint-chunked CE           8.59 GB
-> motivated the sequenced custom-VJP rewrite measured by 'delta'/'ce' below.
"""

import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn

import train_runner

T = 8192
GB = 1024**3


def report(name):
    mx.eval()
    print(f"{name}: peak {mx.get_peak_memory() / GB:.2f} GB")


def exp_sdpa():
    # One full-attention layer's core: 16 q heads / 4 kv heads, head_dim 256
    q = mx.random.normal((1, 16, T, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, T, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 4, T, 256)).astype(mx.bfloat16)

    def f(q, k, v):
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
        return o.astype(mx.float32).sum()

    val, grads = mx.value_and_grad(f, argnums=(0, 1, 2))(q, k, v)
    mx.eval(val, *grads)
    report("sdpa fwd+bwd T=8192")


def exp_delta():
    # One GatedDeltaNet recurrence, real dims, sequenced custom VJP
    B, Hk, Hv, Dk, Dv = 1, 16, 32, 128, 128
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,)) * 0.1
    dt_bias = mx.random.normal((Hv,)) * 0.1

    patched = train_runner._make_patched_gated_delta_update()

    def f(q, k, v, a, b):
        y, s = patched(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=False)
        return y.astype(mx.float32).sum() + s.sum()

    val, grads = mx.value_and_grad(f, argnums=(0, 1, 2, 3, 4))(q, k, v, a, b)
    mx.eval(val, *grads)
    report("delta sequenced (1 layer) T=8192")


def exp_ce():
    V, H = 248320, 4096
    w = mx.random.normal((V, H)).astype(mx.bfloat16) * 0.01
    h = mx.random.normal((1, T, H)).astype(mx.bfloat16)
    t = mx.random.randint(0, V, (1, T))
    m = mx.ones((1, T), dtype=mx.bool_)

    def head(h_c):
        return h_c @ w.T

    def f(h):
        return train_runner._sequenced_ce_sum(head, h, t, m)

    val, grad = mx.value_and_grad(f)(h)
    mx.eval(val, grad)
    report("ce sequenced T=8192 V=248320")


EXPERIMENTS = {
    "sdpa": exp_sdpa,
    "delta": exp_delta,
    "ce": exp_ce,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mx.random.seed(3)
        EXPERIMENTS[sys.argv[1]]()
    else:
        for name in EXPERIMENTS:
            r = subprocess.run([sys.executable, __file__, name])
            if r.returncode != 0:
                print(f"{name}: FAILED (exit {r.returncode})")
