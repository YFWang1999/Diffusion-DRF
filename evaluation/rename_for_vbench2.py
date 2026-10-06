"""Copy sampled videos to the file names expected by VBench-2.0: ``<prompt>-<index>.mp4``.

``inference/predict_t2v.py`` names videos by the line index of the prompt file. This script
pairs the sorted videos of one sampling run with the lines of a prompt list (the original,
non-augmented VBench-2.0 prompts, in the same order as the prompts used for sampling) and
copies them under the VBench-2.0 naming scheme. Run it once per sample index.
"""
import argparse
import os
import shutil


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace(":", "-").replace("?", "").replace("*", "")
    name = name.replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    return name or "renamed_file"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="Output folder of one predict_t2v.py run.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt list that defines the file names.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_index", type=int, required=True, help="0, 1, 2, ... for repeated samples.")
    parser.add_argument("--max_len", type=int, default=180, help="Truncate prompts to this many characters.")
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    videos = sorted(v for v in os.listdir(args.video_dir) if v.lower().endswith(".mp4"))
    if len(videos) != len(prompts):
        raise ValueError(f"{len(videos)} videos in {args.video_dir} but {len(prompts)} prompts in {args.prompt_file}.")

    os.makedirs(args.output_dir, exist_ok=True)
    for video, prompt in zip(videos, prompts):
        target = os.path.join(args.output_dir, f"{sanitize_filename(prompt)[:args.max_len]}-{args.sample_index}.mp4")
        if not os.path.exists(target):
            shutil.copy2(os.path.join(args.video_dir, video), target)
    print(f"Copied {len(videos)} videos to {args.output_dir}")


if __name__ == "__main__":
    main()
