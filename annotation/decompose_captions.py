"""Step 1 of the annotation pipeline: decompose video captions into atomic scene elements.

A text-only LLM (Qwen2.5-7B-Instruct by default) lists, for each caption, what the prompt
explicitly asks for in eight categories (environment, objects, motion, layout, colour,
lighting, text, camera motion).

Input : a tab-separated file with the columns ``video_id``, ``real_video_path``, ``prompt``.
Output: a JSON file ``{video_id: {category: description}}``.
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import CATEGORIES, build_decomposition_prompt


def parse_decomposition(text: str):
    """Extract the JSON object from the model response; ``None`` if it cannot be parsed."""
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return {c: str(parsed[c]).strip() for c in CATEGORIES if c in parsed}


@torch.no_grad()
def decompose(model, tokenizer, caption: str, max_new_tokens: int = 512):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": build_decomposition_prompt(caption)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_decomposition(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True,
                        help="TSV with columns video_id, real_video_path, prompt.")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the captions for multi-GPU runs.")
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_csv(args.captions, sep="\t", encoding="utf-8-sig")
    data = data.drop_duplicates(subset=["video_id"]).iloc[args.shard_index::args.num_shards]

    results = {}
    if os.path.exists(args.output_json):  # resume
        with open(args.output_json, "r", encoding="utf-8") as f:
            results = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto").eval()

    for i, row in enumerate(tqdm(data.itertuples(index=False), total=len(data))):
        if row.video_id in results:
            continue
        decomposition = decompose(model, tokenizer, row.prompt)
        if decomposition is None:
            print(f"Could not parse the decomposition of {row.video_id}; skipping.")
            continue
        results[row.video_id] = decomposition
        if (i + 1) % 100 == 0:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(results)} decompositions to {args.output_json}")


if __name__ == "__main__":
    main()
