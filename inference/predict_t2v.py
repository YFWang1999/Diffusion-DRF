"""Text-to-video sampling with Wan2.1-T2V, optionally loading a Diffusion-DRF checkpoint.

One video is written per line of the prompt file as ``<save_dir>/<8-digit index>.mp4``.
The defaults reproduce the sampling setup used for the evaluations in the paper
(512x288, 49 frames, 30 steps, guidance 6.0, shift 3, TeaCache threshold 0.10).
"""
import argparse
import os
import sys

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel,
                               WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.utils import filter_kwargs, save_videos_grid

# Default negative prompt of Wan2.1.
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the Wan2.1 T2V weights.")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Fine-tuned transformer weights, e.g. "
                             "<output_dir>/checkpoint-1500/transformer/diffusion_pytorch_model.safetensors. "
                             "Omit to sample from the base model.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Text file with one prompt per line.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_videos", type=int, default=-1, help="Only use the first N prompts (-1: all).")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--sampler_name", type=str, default="Flow", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--seed", type=int, default=42, help="-1 draws fresh noise for every video.")
    parser.add_argument("--teacache_threshold", type=float, default=0.10, help="0 disables TeaCache.")
    parser.add_argument("--teacache_model_key", type=str, default="wan2.1-t2v-1.3b",
                        help="Selects the TeaCache coefficients.")
    parser.add_argument("--gpu_memory_mode", type=str, default="sequential_cpu_offload",
                        choices=["sequential_cpu_offload", "model_cpu_offload", "model_full_load"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    config = OmegaConf.load(args.config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.model_dir, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    if args.transformer_path is not None:
        print(f"Loading transformer weights from {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_dir, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_dir, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = WanPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
    )
    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if args.teacache_threshold > 0:
        coefficients = get_teacache_coefficients(args.teacache_model_key)
        if coefficients is not None:
            print(f"TeaCache enabled with threshold {args.teacache_threshold}.")
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, args.teacache_threshold, num_skip_start_steps=5, offload=False,
            )

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.max_videos > 0:
        prompts = prompts[:args.max_videos]

    # A single generator is shared by all prompts of a run, so a video depends on the seed
    # and on its position in the prompt file.
    generator = None if args.seed == -1 else torch.Generator(device=device).manual_seed(args.seed)
    temporal_ratio = vae.config.temporal_compression_ratio
    video_length = int((args.video_length - 1) // temporal_ratio * temporal_ratio) + 1 if args.video_length != 1 else 1

    os.makedirs(args.save_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        video_path = os.path.join(args.save_dir, f"{i + 1:08d}.mp4")
        if os.path.exists(video_path):
            continue
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                shift=args.shift,
            ).videos
        save_videos_grid(sample, video_path, fps=args.fps)
        print(f"[{i + 1}/{len(prompts)}] saved {video_path}")


if __name__ == "__main__":
    main()
