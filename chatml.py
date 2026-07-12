"""Manual ChatML rendering matching andjiang/CoPaw-Flash-9B-oQ4's chat template.

The model's own `chat_template.jinja` raises `TemplateError: No user query
found in messages` whenever a message list contains no genuine user turn
(only tool/assistant turns) -- e.g. Plan v2's segment.py deliberately creates
episodes that start mid-trajectory at an assistant "summary" turn, with no
leading user message at all. That's legitimate data, not a formatting bug,
so instead of dropping those episodes or fabricating a fake user turn, this
module reimplements the template's actual rendering algorithm directly in
Python -- verified to produce byte-identical output to
`tokenizer.apply_chat_template` on every case the template itself accepts
(see the exact TemplateError-triggering check it replicates: a message list
needs a "real" user message, i.e. not one wrapping a `<tool_response>`, for
the last-turn <think> block to have somewhere to anchor -- our data never
puts tool results under role="user" in the first place, so that distinction
never applies here; every role="user" message is a genuine query).

Used both to build training text (train_sft.py) and eval-time generation
prompts (eval_completion.py), so the two stay byte-for-byte consistent.
"""

from __future__ import annotations


def render_chatml(
    messages: list[dict],
    system_prompt: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n")

    last_query_index = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_query_index = i
            break

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = (msg.get("content") or "").strip()
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            if i > last_query_index:
                parts.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            is_last = i == len(messages) - 1
            next_is_tool = (not is_last) and messages[i + 1]["role"] == "tool"
            if is_last or not next_is_tool:
                parts.append("<|im_end|>\n")
        else:
            raise ValueError(f"Unexpected role: {role!r}")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n<think>\n")

    return "".join(parts)
