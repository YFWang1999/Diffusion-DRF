"""Step 2 of the annotation pipeline: build structured queries and reference answers.

For every captioned real video this script
  * turns each decomposed component into a text-video alignment (TA) query and answers it
    with a frozen VLM on the *real* video, which gives the reference answer (a Yes/No decision
    followed by a short explanation);
  * adds one reference-guided physics (Phy) query whose reference answer is fixed.

The visual-quality (VQ) query is the same for every sample and is added at training time by
``--mix_loss``. The output is the tab-separated file consumed by ``scripts/train_drf.py``
(one row per query), see ``data/README.md``.
"""
import argparse
import csv
import json
import os
import re

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from prompts import (CATEGORIES, NOT_STATED, PHY_REFERENCE_ANSWER,
                     build_phy_question, build_ta_question)

FIELDS = ["video_id", "real_video_path", "prompt", "questions", "gt_answers"]


@torch.no_grad()
def answer_on_video(model, processor, video_path: str, question: str, num_frames: int, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": num_frames},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs
    ).to(model.device)
    # The model's default generation config is used (near-greedy: top_k=1).
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def strip_code_fence(answer: str) -> str:
    """Remove a surrounding ```json ... ``` fence so that the reference answer is plain JSON."""
    answer = answer.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", answer, re.DOTALL)
    return match.group(1) if match else answer


def is_yes(answer: str) -> bool:
    match = re.search(r'"answer"\s*:\s*"?(\w+)', answer)
    return match is not None and match.group(1).lower() == "yes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True,
                        help="TSV with columns video_id, real_video_path, prompt.")
    parser.add_argument("--decompositions", type=str, required=True, help="Output of decompose_captions.py.")
    parser.add_argument("--video_root", type=str, default="", help="Prefix joined to real_video_path.")
    parser.add_argument("--output", type=str, required=True, help="Annotation TSV to write.")
    parser.add_argument("--vlm_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames of the real video shown to the VLM.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--keep_all_answers", action="store_true",
                        help="Keep TA queries whose reference answer is `No`. By default only queries "
                             "answered `Yes` on the real video are kept, as in the released annotations.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_csv(args.captions, sep="\t", encoding="utf-8-sig")
    data = data.drop_duplicates(subset=["video_id"]).iloc[args.shard_index::args.num_shards]
    with open(args.decompositions, "r", encoding="utf-8") as f:
        decompositions = json.load(f)

    processor = AutoProcessor.from_pretrained(args.vlm_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype="auto", device_map="auto"
    ).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    num_queries = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in tqdm(data.itertuples(index=False), total=len(data)):
            decomposition = decompositions.get(row.video_id)
            if decomposition is None:
                continue
            video_path = os.path.join(args.video_root, row.real_video_path)
            queries = []
            for category in CATEGORIES:
                description = decomposition.get(category, NOT_STATED).strip()
                if not description or description.rstrip(".").lower() == NOT_STATED.rstrip(".").lower():
                    continue
                question = build_ta_question(category, description)
                answer = strip_code_fence(
                    answer_on_video(model, processor, video_path, question, args.num_frames, args.max_new_tokens)
                )
                if args.keep_all_answers or is_yes(answer):
                    queries.append((question, answer))
            if not queries:
                continue
            queries.append((build_phy_question(row.prompt), PHY_REFERENCE_ANSWER))

            for question, answer in queries:
                writer.writerow({
                    "video_id": row.video_id,
                    "real_video_path": row.real_video_path,
                    "prompt": row.prompt,
                    "questions": repr([question]),
                    "gt_answers": repr([answer]),
                })
                num_queries += 1
            f.flush()
    print(f"Wrote {num_queries} queries to {args.output}")


if __name__ == "__main__":
    main()
