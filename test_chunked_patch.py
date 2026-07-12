"""Numerical equivalence tests for train_runner's manual-BPTT engine.

Run with the training venv python. Small shapes only -- verifies math, not
memory. Exits nonzero on any mismatch.
"""

import sys

import mlx.core as mx
import mlx.nn as nn

import train_runner

PASS = True


def check(name, a, b, atol=1e-4, rtol=1e-3):
    global PASS
    a = mx.array(a).astype(mx.float32)
    b = mx.array(b).astype(mx.float32)
    ok = bool(mx.allclose(a, b, atol=atol, rtol=rtol).item())
    diff = float(mx.abs(a - b).max().item()) if a.size else 0.0
    print(f"{'PASS' if ok else 'FAIL'} {name} (max abs diff {diff:.3e})")
    PASS = PASS and ok


def test_chunkwise_scan():
    """Chunkwise-parallel scan vs exact per-timestep reference: forward,
    final state, and all input gradients, at real head dims."""
    mx.random.seed(5)
    B, C, H, Dk, Dv = 1, 128, 32, 128, 128
    st = mx.random.normal((B, H, Dv, Dk)).astype(mx.float32) * 0.3
    # fp32 inputs: the per-step reference rounds y to the input dtype every
    # step, so bf16 comparisons measure reference rounding, not our error.
    q = mx.random.normal((B, C, H, Dk)) * 0.2
    k = mx.random.normal((B, C, H, Dk)) * 0.2
    v = mx.random.normal((B, C, H, Dv)) * 0.5
    g = mx.sigmoid(mx.random.normal((B, C, H)) - 1.0)  # incl. strong decays
    beta = mx.sigmoid(mx.random.normal((B, C, H)))
    dy = mx.random.normal((B, C, H, Dv))
    dstate = mx.random.normal((B, H, Dv, Dk)).astype(mx.float32)

    def via(scan_fn):
        def f(st, q, k, v, g, beta):
            return scan_fn(st, q, k, v, g, beta)

        outs, grads = mx.vjp(f, [st, q, k, v, g, beta], [dy, dstate])
        mx.eval(*outs, *grads)
        return outs, grads

    (ry, rs), rgrads = via(lambda *a: train_runner._scan_chunk(*a, None))
    (fy, fs), fgrads = via(train_runner._chunk_scan_fast)

    check("chunkwise y", ry, fy, atol=2e-3, rtol=2e-2)
    check("chunkwise state", rs, fs, atol=2e-3, rtol=2e-2)
    for name, r, f_ in zip(["dst", "dq", "dk", "dv", "dg", "dbeta"], rgrads, fgrads):
        check(f"chunkwise grad {name}", r, f_, atol=3e-3, rtol=3e-2)

    # Padding no-op semantics: g=1, beta=0 must pass state through untouched
    g1 = mx.ones((B, 8, H))
    b0 = mx.zeros((B, 8, H))
    _, s_pass = train_runner._chunk_scan_fast(st, q[:, :8], k[:, :8], v[:, :8], g1, b0)
    mx.eval(s_pass)
    check("pad passthrough state", st, s_pass, atol=1e-6)


def build_tiny_lora_model():
    from mlx_lm.models import qwen3_5
    from mlx_lm.tuner.utils import linear_to_lora_layers

    text_config = dict(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,  # 6 linear + 2 full attention
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,  # keeps default mrope_section valid (sum 32 == rotary/2)
        vocab_size=96,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        # >=32: the fused kernel needs head_k_dim/32 threads per lane
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        tie_word_embeddings=False,
    )
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text_config)
    mx.random.seed(11)
    model = qwen3_5.Model(args)
    model.freeze()
    linear_to_lora_layers(
        model, 8, {"rank": 4, "scale": 1.0, "dropout": 0.0}
    )
    model.train()
    mx.eval(model.parameters())
    return model


def test_manual_engine():
    from mlx.utils import tree_flatten

    from mlx_lm.tuner.trainer import default_loss

    train_runner.CE_CHUNK = 8
    train_runner.DELTA_CHUNK = 16
    train_runner.install_all_patches()

    model = build_tiny_lora_model()
    batch = mx.random.randint(0, 96, (1, 41))
    lengths = mx.array([[3, 40]])

    # Reference: stock loss + autodiff (training mode -> exact ops recurrence)
    ref_loss = nn.value_and_grad(model, default_loss)
    (ref_val, ref_toks), ref_grad = ref_loss(model, batch, lengths)
    mx.eval(ref_val, ref_grad)
    ref_flat = dict(tree_flatten(ref_grad))

    # Manual engine
    man_val, man_toks, man_grads = train_runner.manual_loss_and_grads(
        model, batch, lengths
    )

    check("loss value", ref_val, man_val, atol=5e-4)
    check("ntoks", ref_toks, man_toks)

    assert set(ref_flat) == set(man_grads), (
        "grad key mismatch:\n"
        f"only ref: {sorted(set(ref_flat) - set(man_grads))[:5]}\n"
        f"only manual: {sorted(set(man_grads) - set(ref_flat))[:5]}"
    )
    worst = max(
        ref_flat,
        key=lambda k: float(mx.abs(ref_flat[k] - man_grads[k]).max().item()),
    )
    check(
        f"grads ({len(ref_flat)} leaves; worst: {worst})",
        ref_flat[worst],
        man_grads[worst],
        atol=2e-3,
        rtol=5e-3,
    )

    # Eval loss (kernel path) must match default_loss value in eval mode
    model.eval()
    (ref_eval, _), = [default_loss(model, batch, lengths)]
    ev, _ = train_runner.eval_chunked_loss(model, batch, lengths)
    mx.eval(ref_eval)
    check("eval loss value", ref_eval, ev, atol=5e-4)
    model.train()

    # Guard fires when the head is trainable
    model.language_model.lm_head.unfreeze()
    try:
        train_runner._assert_head_frozen(model.language_model)
        print("FAIL frozen-head guard did not fire")
        global PASS
        PASS = False
    except RuntimeError:
        print("PASS frozen-head guard fires on trainable head")


if __name__ == "__main__":
    test_chunkwise_scan()
    test_manual_engine()
    print("ALL PASS" if PASS else "FAILURES")
    sys.exit(0 if PASS else 1)
