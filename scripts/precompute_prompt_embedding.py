"""
Precompute fixed-prompt embedding for TADiSR and save to disk.

Usage:
  python scripts/precompute_prompt_embedding.py \
      --pretrained_path Kwai-Kolors/Kolors-diffusers \
      --prompt "A high-quality photo with clear text" \
      --output assets/fixed_prompt_context.pt
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str,
                        default="Kwai-Kolors/Kolors-diffusers")
    parser.add_argument("--prompt", type=str,
                        default="A high-quality photo with clear text")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_path, subfolder="text_encoder", trust_remote_code=True
    )
    text_encoder = AutoModel.from_pretrained(
        args.pretrained_path, subfolder="text_encoder", trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    text_encoder.eval()

    with torch.no_grad():
        inputs = tokenizer(
            args.prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )
        outputs = text_encoder(**inputs)
        context = outputs.last_hidden_state.cpu()  # [1, S, C]

    tokens = tokenizer.tokenize(args.prompt)
    text_token_indices = [i for i, tok in enumerate(tokens) if "text" in tok.lower()]
    if not text_token_indices:
        text_token_indices = [len(tokens) - 1]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "prompt": args.prompt,
            "context": context,
            "text_token_indices": text_token_indices,
            "tokenizer_tokens": tokens,
        },
        args.output
    )

    print(f"Saved precomputed prompt context to: {args.output}")
    print(f"context shape: {tuple(context.shape)}")
    print(f"text_token_indices: {text_token_indices}")


if __name__ == "__main__":
    main()
