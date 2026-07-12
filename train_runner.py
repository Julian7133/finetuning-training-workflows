"""Memory-safe wrapper around `mlx_lm.lora` for training qwen3_5-style models.

Why this exists (instead of calling `python -m mlx_lm lora` directly):

The model (CoPaw-Flash-9B, qwen3_5) has 24/32 GatedDeltaNet linear-attention
layers. In training mode mlx_lm's fused Metal kernel has no VJP, so it falls
back to `gated_delta_ops`: a per-timestep loop whose fp32 recurrent state
(B, 32, 128, 128) = 2.1MB/step is needed by the backward pass. One layer's
backward at 4096 tokens is ~20GB+, which is why the observed peak was
independent of max_seq_length and killed every run on this 24GB M3.

Graph-level fixes DO NOT work (all measured, see diag_mem.py):
- `mx.checkpoint` chunking: MLX runs recomputes concurrently -> no reduction.
- mlx_lm's per-layer `grad_checkpoint`: same problem (hence "no difference").
- custom VJP with `mx.depends` sequencing: buffers are still allocated for
  the whole encoded graph; peak grew linearly with T (66GB at T=8192 for ONE
  layer) regardless of chunk size, memory limit, or MLX_MAX_OPS_PER_BUFFER.

The only reliable memory boundary in MLX is an explicit `mx.eval()` between
Python-level segments. So this runner implements MANUAL backprop with eval
boundaries instead of one big value_and_grad graph:

  Phase F: forward layer by layer (recurrence via the fast inference kernel,
           valid because no grads are needed here), mx.eval per layer, saving
           each layer's input (32 x ~67MB bf16 at 8k tokens).
  Phase C: final-norm + lm_head + CE per CE_CHUNK tokens; value_and_grad per
           chunk + mx.eval -> loss and d(hidden) without ever materializing
           the full (L, 248320) logits (~4GB bf16 at 8k + same-sized grad).
  Phase B: layers in reverse; full-attention/MLP layers via one mx.vjp each;
           GatedDeltaNet layers split into pre-scan / scan / post-scan, with
           true chunked BPTT over the recurrence (DELTA_CHUNK timesteps per
           mx.vjp + mx.eval; boundary states recomputed with the kernel).

Only LoRA gradients (tiny) and layer inputs survive segment boundaries, so
peak ~= weights + layer inputs + the single largest segment (the SDPA
backward of one full-attention layer: ~7GB at 8k tokens).

Also: `train()` would wire `max_recommended_working_set_size`, which after
the sysctl iogpu.wired_limit_mb bump can wire ~22GB -- the config that froze
this machine before. We clamp mx.set_wired_limit instead of touching sysctl.

Numerical equivalence vs stock mlx_lm is covered by test_chunked_patch.py.
The output head must be frozen (LoRA): its gradient is not computed.

Usage: python train_runner.py -c output/lora_config.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

GB = 1024**3

# Hard cap on what may be wired: comfortably below the oMLX hard threshold
# (19.9GB) and the sysctl-raised iogpu wired limit (22GB).
WIRED_CAP_BYTES = int(14.0 * GB)

CACHE_LIMIT_BYTES = int(2 * GB)

# Timesteps per BPTT chunk of the deltanet recurrence. Peak per chunk ~=
# DELTA_CHUNK * ~3 * 2.1MB fp32 state-sized buffers.
DELTA_CHUNK = 128

# Tokens per CE chunk: logits are CE_CHUNK x 248320 (bf16) + fp32 softmax.
CE_CHUNK = 1024

# Set during no-grad forward passes so GatedDeltaNet uses the fused kernel
# even though module.training is True.
_FORCE_KERNEL = False


def _install_wired_limit_clamp():
    orig = mx.set_wired_limit

    def capped(limit):
        return orig(min(int(limit), WIRED_CAP_BYTES))

    mx.set_wired_limit = capped


def _install_gated_delta_patch():
    """Route training-mode scans to the kernel while _FORCE_KERNEL is set
    (phase F / boundary recomputes); otherwise stock behavior."""
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models import gated_delta as gd

    orig = gd.gated_delta_update

    def gated_delta_update(*args, use_kernel=True, **kwargs):
        return orig(*args, use_kernel=use_kernel or _FORCE_KERNEL, **kwargs)

    q35.gated_delta_update = gated_delta_update


# ---------------------------------------------------------------------------
# Manual backprop engine
# ---------------------------------------------------------------------------


def _scan_chunk(state, q_c, k_c, v_c, g_c, beta_c, mask_c=None):
    """Stock per-timestep recurrence over one chunk (exact ops math)."""
    from mlx_lm.models import gated_delta as gd

    ys = []
    for t in range(q_c.shape[1]):
        m = None if mask_c is None else mask_c[:, t]
        y, state = gd._gated_delta_step_ops(
            q_c[:, t], k_c[:, t], v_c[:, t], g_c[:, t], beta_c[:, t], state, m
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


def _tri_inv_neumann(A):
    """(I + A)^-1 for strictly-lower-triangular A via Neumann doubling:
    A is nilpotent (A^C = 0), so log2(C) squarings give the exact inverse."""
    C = A.shape[-1]
    eye = mx.eye(C, dtype=A.dtype)
    N = eye
    P = -A
    m = 1
    while m < C:
        N = N + P @ N
        P = P @ P
        m *= 2
    return N


def _chunk_scan_fast(state, q_c, k_c, v_c, g_c, beta_c):
    """Chunkwise-parallel gated delta rule (UT transform), mathematically
    identical to _scan_chunk (per-timestep reference):

        S_t = g_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,  y_t = S_t q_t

    With G_t = prod_{s<=t} g_s and pseudo-values u solving the unit lower
    triangular system (I + A) U = Bv,
        A[t,s] = b_t (G_t/G_s)(k_t . k_s)          (s < t)
        Bv[t]  = b_t v_t - b_t G_t (S_0 k_t)
    the states are S_t = G_t S_0 + sum_{s<=t} (G_t/G_s) u_s k_s^T, giving
        y   = G * (Q S_0^T) + (M o incl_tril) U,   M[t,s] = (G_t/G_s)(q_t.k_s)
        S_C = G_C S_0 + K^T diag(G_C/G_s) U  (transposed appropriately)

    All matmuls -> a handful of GPU ops instead of C sequential steps, and
    autodiff through this gives the backward as matmuls too. Decay ratios use
    exp(L_t - L_s) with L = cumsum(log g) for stability (ratios <= 1).
    """
    in_dtype = q_c.dtype
    C = q_c.shape[1]

    # (B, C, H, D) -> (B, H, C, D), fp32
    qh = q_c.transpose(0, 2, 1, 3).astype(mx.float32)
    kh = k_c.transpose(0, 2, 1, 3).astype(mx.float32)
    vh = v_c.transpose(0, 2, 1, 3).astype(mx.float32)
    bh = beta_c.transpose(0, 2, 1).astype(mx.float32)  # (B, H, C)
    L = mx.cumsum(mx.log(g_c.astype(mx.float32)), axis=1).transpose(0, 2, 1)

    G = mx.exp(L)  # (B, H, C), G_t in (0, 1]

    # exp(L_t - L_s) is meaningful only for t >= s; the upper triangle can
    # overflow to inf (L decreasing), and inf * 0-mask = NaN. Mask the
    # EXPONENT, not the product.
    diff = L[..., :, None] - L[..., None, :]
    neg_inf = mx.array(-mx.inf, dtype=mx.float32)
    strict_b = mx.tri(C, k=-1, dtype=mx.bool_)
    incl_b = mx.tri(C, k=0, dtype=mx.bool_)
    ratio_strict = mx.exp(mx.where(strict_b, diff, neg_inf))
    ratio_incl = mx.exp(mx.where(incl_b, diff, neg_inf))

    KKt = kh @ kh.transpose(0, 1, 3, 2)  # (B, H, C, C) k_t . k_s
    A = bh[..., None] * ratio_strict * KKt

    S0T = state.transpose(0, 1, 3, 2)  # (B, H, Dk, Dv)
    Bv = bh[..., None] * (vh - G[..., None] * (kh @ S0T))  # (B, H, C, Dv)

    U = _tri_inv_neumann(A) @ Bv  # (B, H, C, Dv)

    QKt = qh @ kh.transpose(0, 1, 3, 2)  # q_t . k_s
    y = G[..., None] * (qh @ S0T) + (ratio_incl * QKt) @ U  # (B, H, C, Dv)

    w = mx.exp(L[..., -1:] - L)  # G_C / G_s, (B, H, C)
    S_out = G[..., -1, None, None] * state + (U * w[..., None]).transpose(
        0, 1, 3, 2
    ) @ kh  # (B, H, Dv, Dk)

    return y.transpose(0, 2, 1, 3).astype(in_dtype), S_out


def _scan_chunk_grads(st, q_c, k_c, v_c, g_c, b_c, dy_c, dstate):
    def f(st, q_c, k_c, v_c, g_c, b_c):
        return _scan_chunk(st, q_c, k_c, v_c, g_c, b_c, None)

    _, grads = mx.vjp(f, [st, q_c, k_c, v_c, g_c, b_c], [dy_c, dstate])
    return tuple(grads)


# All BPTT chunks share one fixed shape (sequences are padded to a multiple
# of DELTA_CHUNK), so this compiles once and removes the dominant cost of the
# manual engine: rebuilding the 128-step chunk graph in Python for every
# chunk of every layer of every iteration.
_scan_chunk_grads_compiled = mx.compile(_scan_chunk_grads)


def _pad_time(a, pad, value=0):
    if pad == 0:
        return a
    widths = [(0, 0), (0, pad)] + [(0, 0)] * (a.ndim - 2)
    return mx.pad(a, widths, constant_values=value)


_ce_grad_fn = None


def _get_ce_grad_fn(lm):
    """Compiled per-chunk CE value+grad. Head/norm weights are frozen, so
    capturing them as compile-time constants is safe."""
    global _ce_grad_fn
    if _ce_grad_fn is None:
        head = _head_fn(lm)
        norm = lm.model.norm

        def f(h_c, t_c, m_c):
            logits = head(norm(h_c))
            ce = nn.losses.cross_entropy(logits, t_c) * m_c
            return ce.astype(mx.float32).sum()

        _ce_grad_fn = mx.compile(mx.value_and_grad(f))
    return _ce_grad_fn


def _flat_trainables(module):
    from mlx.utils import tree_flatten

    flat = tree_flatten(module.trainable_parameters())
    return [k for k, _ in flat], [v for _, v in flat]


def _with_params(module, keys, params):
    from mlx.utils import tree_unflatten

    if keys:
        module.update(tree_unflatten(list(zip(keys, params))))


def _delta_layer_backward(layer, x, dout, keys, params):
    """Backward through one GatedDeltaNet DecoderLayer with chunked BPTT.

    layer(x) == g2(x, scan(g1(x))) where g1 = everything before the
    recurrence (projections, conv, q/k norms, gates), scan = the recurrence,
    g2 = gated norm + out_proj + residual + MLP (recomputes z internally so
    nothing is double-counted). Assumes ssm mask is None (training, no cache,
    right-padding handled by the CE mask, identical to stock behavior).
    """
    from mlx_lm.models import gated_delta as gd

    attn = layer.linear_attn
    B = x.shape[0]
    Hk, Hv = attn.num_k_heads, attn.num_v_heads
    Dk, Dv = attn.head_k_dim, attn.head_v_dim

    def g1(x, *p):
        _with_params(layer, keys, list(p))
        u = layer.input_layernorm(x)
        S = u.shape[1]
        qkv = attn.in_proj_qkv(u)
        b_ = attn.in_proj_b(u)
        a_ = attn.in_proj_a(u)
        conv_state = mx.zeros(
            (B, attn.conv_kernel_size - 1, attn.conv_dim), dtype=u.dtype
        )
        conv_out = nn.silu(attn.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
                [Hk, Hk, Hv],
                [Dk, Dk, Dv],
            )
        ]
        inv_scale = Dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b_)
        g = gd.compute_g(attn.A_log, a_, attn.dt_bias)
        if (rf := Hv // Hk) > 1:
            q = mx.repeat(q, rf, -2)
            k = mx.repeat(k, rf, -2)
        return q, k, v, g, beta

    def g2(x, y, *p):
        _with_params(layer, keys, list(p))
        u = layer.input_layernorm(x)
        S = u.shape[1]
        z = attn.in_proj_z(u).reshape(B, S, Hv, Dv)
        o = attn.norm(y, z)
        r = attn.out_proj(o.reshape(B, S, -1))
        h = x + r
        return h + layer.mlp(layer.post_attention_layernorm(h))

    # Forward values for the scan (no grads yet). Pad the time axis to a
    # multiple of DELTA_CHUNK so every BPTT chunk has the same shape and hits
    # one compiled graph. Pad semantics keep the recurrence a no-op: g=1 (no
    # decay), beta=0 (no update) -> the state passes through padded steps
    # unchanged in both directions.
    q, k, v, g, beta = g1(x, *params)
    T = q.shape[1]
    pad = (-T) % DELTA_CHUNK
    q, k, v, beta = (_pad_time(a, pad) for a in (q, k, v, beta))
    g = _pad_time(g, pad, value=1)
    mx.eval(q, k, v, g, beta)

    Tp = T + pad
    starts = list(range(0, Tp, DELTA_CHUNK))

    # y for g2's backward: ops scan (matches phase F on manual_linear layers).
    state0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    y_parts = []
    st = state0
    for s in starts:
        e = s + DELTA_CHUNK
        y_c, st = _scan_chunk(
            st, q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], None
        )
        y_parts.append(y_c)
    y = mx.concatenate(y_parts, axis=1)[:, :T]

    bounds = [state0]
    st = state0
    for s in starts[:-1]:
        e = s + DELTA_CHUNK
        _, st = _scan_chunk(
            st, q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], None
        )
        bounds.append(st)
    mx.eval(y, *bounds)

    # Backward through g2
    _, grads2 = mx.vjp(g2, [x, y, *params], [dout])
    dx = grads2[0]
    dy = grads2[1]
    dp2 = grads2[2:]
    mx.eval(dx, dy, *dp2)
    dy = _pad_time(dy, pad)

    # Chunked BPTT through the recurrence: one chained matmul graph, one eval.
    dstate = mx.zeros_like(state0)
    parts = []
    for idx in reversed(range(len(starts))):
        s = starts[idx]
        e = s + DELTA_CHUNK
        gr = _scan_chunk_grads_compiled(
            bounds[idx],
            q[:, s:e],
            k[:, s:e],
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            dy[:, s:e],
            dstate,
        )
        dstate = gr[0]
        parts.append(gr[1:])
    parts = parts[::-1]
    dq, dk, dv, dg, dbeta = (
        mx.concatenate([p[i] for p in parts], axis=1)[:, :T] for i in range(5)
    )
    mx.eval(dq, dk, dv, dg, dbeta)

    # Backward through g1
    _, grads1 = mx.vjp(g1, [x, *params], [dq, dk, dv, dg, dbeta])
    dx = dx + grads1[0]
    dp = [a + b for a, b in zip(grads1[1:], dp2)]
    mx.eval(dx, *dp)
    return dx, dp


def _attention_layer_backward(layer, x, dout, keys, params, fa_mask):
    def f(x, *p):
        _with_params(layer, keys, list(p))
        return layer(x, mask=fa_mask, cache=None)

    _, grads = mx.vjp(f, [x, *params], [dout])
    mx.eval(*grads)
    return grads[0], list(grads[1:])


def _assert_head_frozen(lm):
    from mlx.utils import tree_flatten

    head = lm.model.embed_tokens if lm.args.tie_word_embeddings else lm.lm_head
    if tree_flatten(head.trainable_parameters()):
        raise RuntimeError(
            "train_runner requires a frozen output head (LoRA); the head has "
            "trainable parameters whose gradients would be silently dropped."
        )
    if tree_flatten(lm.model.embed_tokens.trainable_parameters()) or tree_flatten(
        lm.model.norm.trainable_parameters()
    ):
        raise RuntimeError(
            "train_runner requires frozen embeddings and final norm (LoRA)."
        )


def _ce_mask(targets, lengths):
    steps = mx.arange(1, targets.shape[1] + 1)
    return mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])


def _head_fn(lm):
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear
    return lm.lm_head


def manual_loss_and_grads(model, batch, lengths):
    """Returns (mean_loss, ntoks, grads) where grads is keyed by full
    parameter paths (matching model.trainable_parameters() layout)."""
    global _FORCE_KERNEL

    lm = model.language_model
    _assert_head_frozen(lm)
    tm = lm.model

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    L = targets.shape[1]
    fa_mask = "causal" if inputs.shape[1] > 1 else None

    trainable_idx = [
        i for i, l in enumerate(tm.layers) if _flat_trainables(l)[0]
    ]
    lowest = min(trainable_idx) if trainable_idx else len(tm.layers)
    # Phase B backprops through g1/scan/g2 ops math on linear layers. Phase F
    # must use the same recurrence for those layers, not the inference kernel,
    # or CE is evaluated on kernel hidden states while grads follow ops -> NaN.
    manual_linear = {
        i
        for i in range(lowest, len(tm.layers))
        if tm.layers[i].is_linear
    }

    # ---- Phase F: forward, eval-bounded per layer
    h = tm.embed_tokens(inputs)
    mx.eval(h)
    layer_inputs = []
    _FORCE_KERNEL = True
    try:
        for i, layer in enumerate(tm.layers):
            layer_inputs.append(h)
            mask = None if layer.is_linear else fa_mask
            if i in manual_linear:
                _FORCE_KERNEL = False
                h = layer(h, mask=mask, cache=None)
                _FORCE_KERNEL = True
            else:
                h = layer(h, mask=mask, cache=None)
            mx.eval(h)
    finally:
        _FORCE_KERNEL = False

    # ---- Phase C: final norm + head + CE, chunked; produces loss and dh
    ce_mask = _ce_mask(targets, lengths)
    ntoks = ce_mask.sum()
    mx.eval(ntoks)
    inv_ntoks = 1.0 / ntoks.item()
    head = _head_fn(lm)

    # Pad to a multiple of CE_CHUNK (mask=False on pads) so every chunk hits
    # one compiled graph.
    ce_grad = _get_ce_grad_fn(lm)
    pad = (-L) % CE_CHUNK
    h_p = _pad_time(h, pad)
    t_p = _pad_time(targets, pad)
    m_p = _pad_time(ce_mask, pad)

    total = 0.0
    dh_chunks = []
    for s in range(0, L + pad, CE_CHUNK):
        e = s + CE_CHUNK
        part, dh_c = ce_grad(h_p[:, s:e], t_p[:, s:e], m_p[:, s:e])
        dh_c = dh_c * inv_ntoks
        mx.eval(part, dh_c)
        total += part.item()
        dh_chunks.append(dh_c)
    dh = mx.concatenate(dh_chunks, axis=1)[:, :L]
    mx.eval(dh)

    # ---- Phase B: layers in reverse, eval-bounded per layer/chunk.
    # Everything below the lowest layer with trainable params is frozen
    # (including the embedding), so gradients there are useless -- stop.
    grads: dict[str, mx.array] = {}
    for i in reversed(range(lowest, len(tm.layers))):
        layer = tm.layers[i]
        keys, params = _flat_trainables(layer)
        if layer.is_linear:
            dh, dp = _delta_layer_backward(layer, layer_inputs[i], dh, keys, params)
        else:
            dh, dp = _attention_layer_backward(
                layer, layer_inputs[i], dh, keys, params, fa_mask
            )
        prefix = f"language_model.model.layers.{i}."
        for kname, garr in zip(keys, dp):
            grads[prefix + kname] = garr
        # dh of layer 0 would be the (frozen) embedding gradient -- dropped.

    return mx.array(total * inv_ntoks), ntoks, grads


def eval_chunked_loss(model, batch, lengths):
    """Forward-only loss for validation, chunked + eval-bounded so full-vocab
    logits are never materialized. Same semantics as default_loss."""
    lm = model.language_model
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    hidden = lm.model(inputs)  # eval mode -> kernel scans, cheap
    mx.eval(hidden)

    ce_mask = _ce_mask(targets, lengths)
    head = _head_fn(lm)
    # lm.model() already applied the final norm.
    total = 0.0
    L = targets.shape[1]
    for s in range(0, L, CE_CHUNK):
        e = min(s + CE_CHUNK, L)
        ce = nn.losses.cross_entropy(head(hidden[:, s:e]), targets[:, s:e])
        part = (ce * ce_mask[:, s:e]).astype(mx.float32).sum()
        mx.eval(part)
        total += part.item()
    ntoks = ce_mask.sum()
    mx.eval(ntoks)
    return mx.array(total) / ntoks, ntoks


# ---------------------------------------------------------------------------
# Manual training loop (replaces mlx_lm.tuner.trainer.train)
# ---------------------------------------------------------------------------


def manual_train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args=None,
    loss=None,  # ignored -- manual engine computes the loss itself
    iterate_batches=None,
    training_callback=None,
):
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.trainer import evaluate
    from mlx_lm.tuner.trainer import iterate_batches as default_iterate_batches

    iterate_batches = iterate_batches or default_iterate_batches
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print(
        f"Starting training..., iters: {args.iters} "
        f"(train_runner manual-BPTT engine, delta_chunk={DELTA_CHUNK}, ce_chunk={CE_CHUNK})"
    )

    grad_accum_steps = max(1, args.grad_accumulation_steps)
    acc: dict[str, mx.array] | None = None

    losses = 0.0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0.0

    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
        ),
    ):
        if val_dataset and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=eval_chunked_loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            print(
                f"Iter {it}: Val loss {val_loss:.3f}, "
                f"Val took {time.perf_counter() - tic:.3f}s",
                flush=True,
            )
            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it - 1,
                        "val_loss": val_loss,
                        "val_time": time.perf_counter() - tic,
                    }
                )

        tic = time.perf_counter()
        lvalue, toks, grads = manual_loss_and_grads(model, *batch)

        if not (mx.isfinite(lvalue).all().item() and all(
            mx.isfinite(g).all().item() for g in grads.values()
        )):
            raise RuntimeError(
                f"Non-finite loss/grads at iter {it} — stopping before corrupting "
                "adapter weights. Report this with the current batch/seed."
            )

        if acc is None:
            acc = grads
        else:
            acc = {k: acc[k] + grads[k] for k in acc}
            mx.eval(*acc.values())

        if it % grad_accum_steps == 0:
            if grad_accum_steps > 1:
                acc = {k: v / grad_accum_steps for k, v in acc.items()}
            optimizer.update(model, tree_unflatten(list(acc.items())))
            mx.eval(model.trainable_parameters(), optimizer.state)
            acc = None

        losses += lvalue.item()
        n_tokens += toks.item()
        steps += 1
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = losses / steps
            lr = optimizer.learning_rate.item()
            peak_gb = mx.get_peak_memory() / GB
            print(
                f"Iter {it}: Train loss {train_loss:.3f}, "
                f"Learning Rate {lr:.3e}, "
                f"It/sec {steps / train_time:.3f}, "
                f"Tokens/sec {n_tokens / train_time:.3f}, "
                f"Trained Tokens {trained_tokens + n_tokens}, "
                f"Peak mem {peak_gb:.3f} GB",
                flush=True,
            )
            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        "learning_rate": lr,
                        "iterations_per_second": steps / train_time,
                        "tokens_per_second": n_tokens / train_time,
                        "trained_tokens": trained_tokens + n_tokens,
                        "peak_memory": peak_gb,
                    }
                )
            trained_tokens += n_tokens
            losses = 0.0
            n_tokens = 0
            steps = 0
            train_time = 0.0

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")


# ---------------------------------------------------------------------------
# Wiring into mlx_lm.lora
# ---------------------------------------------------------------------------


def _install_trainer_hooks():
    import mlx_lm.lora as lora_mod
    from mlx_lm.tuner.trainer import evaluate as real_evaluate

    def evaluate_with_chunked_loss(**kwargs):
        kwargs.setdefault("loss", eval_chunked_loss)
        return real_evaluate(**kwargs)

    lora_mod.train = manual_train
    lora_mod.evaluate = evaluate_with_chunked_loss


def install_all_patches():
    _install_wired_limit_clamp()
    _install_gated_delta_patch()


def main():
    install_all_patches()
    _install_trainer_hooks()
    mx.set_cache_limit(CACHE_LIMIT_BYTES)

    import mlx_lm.lora as lora_mod

    lora_mod.main()


if __name__ == "__main__":
    sys.exit(main())
