from __future__ import annotations

import argparse

import torch

from minimind_kd.modeling.model import MiniMindKDForCausalLM
from minimind_kd.protocol import effort_prompt
from minimind_kd.training.supervised import resolve_device

from .common import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MiniMind-KD chat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--effort", choices=["none", "high", "max"], default="high")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    model = MiniMindKDForCausalLM.from_pretrained(args.checkpoint).to(device).eval()
    rendered = effort_prompt(args.prompt, args.effort)
    input_ids = torch.tensor([tokenizer.encode(rendered, add_special_tokens=False)], device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = generated[0, input_ids.shape[1] :].tolist()
    print(tokenizer.decode(completion, skip_special_tokens=False))


if __name__ == "__main__":
    main()
