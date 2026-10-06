"""Diffusion-DRF: fine-tune a Wan2.1 text-to-video model with differentiable VLM feedback.

For every training query the script
  1. rolls out the full denoising chain from noise, keeping gradients only for the last
     ``--backprop_num_steps`` steps,
  2. decodes the latent video and sub-samples ``--num_frames`` frames,
  3. feeds the frames (plus reference frames for reference-guided queries) and the
     structured question to a frozen VLM and computes the teacher-forced likelihood of the
     reference answer,
  4. back-propagates that likelihood through the VLM, the decoder and the retained
     denoising steps into the diffusion transformer.

Built on VideoX-Fun (https://github.com/aigc-apps/VideoX-Fun), Apache License 2.0.
"""
import argparse
import ast
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from typing import Optional

import accelerate
import datasets
import decord
import diffusers
import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
import torchvision.transforms as transforms
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from packaging import version
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import videox_fun.reward.reward_fn as reward_fn
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel,
                               WanTransformer3DModel)
from videox_fun.utils.utils import save_videos_grid

logger = get_logger(__name__, log_level="INFO")

# Default negative prompt of Wan2.1.
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

VLM_REWARDS = ("QwenReward", "InternVLReward")


def filter_kwargs(cls, kwargs):
    import inspect

    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 16384, max_pixels: int = 65536):
    """Resolution the VLM sees: both sides divisible by ``factor``, pixel count within
    ``[min_pixels, max_pixels]``, aspect ratio kept as closely as possible (as in qwen-vl-utils)."""
    max_ratio = 200
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def extract_ref_frames(video_path: str, num_frames: int = 1) -> torch.Tensor:
    """Read reference frames from a real video as a uint8 tensor ``[n, H, W, C]``.

    One frame: the middle frame. Several frames: uniformly spaced over the whole clip.
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Reference video {video_path} has no frames.")
    num_frames = min(num_frames, total_frames)
    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = torch.from_numpy(vr.get_batch(indices).asnumpy())
    del vr
    gc.collect()
    return frames


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt,
    negative_prompt,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, negative_prompt_embeds


def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect
    
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-DRF training for Wan2.1.")
    # ---- models -----------------------------------------------------------------------
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Directory of the Wan2.1 T2V weights (original Wan-AI layout).")
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Optional transformer checkpoint (.safetensors/.pth) to start from.")
    parser.add_argument("--reward_decoder", type=str, default="wan", choices=["wan", "lightvae"],
                        help="Decoder used in the differentiable reward path: the official Wan2.1 VAE, "
                             "or LightVAE, a channel-pruned Wan2.1 VAE that needs far less memory.")
    parser.add_argument("--reward_decoder_path", type=str, default=None,
                        help="Weights of the reward decoder when --reward_decoder is not `wan` "
                             "(e.g. lightvaew2_1.safetensors).")
    # ---- data -------------------------------------------------------------------------
    parser.add_argument("--data_path", type=str, required=True,
                        help="Tab-separated annotation file, see data/README.md.")
    parser.add_argument("--video_root", type=str, default="",
                        help="Prefix joined to the `real_video_path` column.")
    # ---- rollout ----------------------------------------------------------------------
    parser.add_argument("--train_sample_height", type=int, default=512)
    parser.add_argument("--train_sample_width", type=int, default=288)
    parser.add_argument("--video_length", type=int, default=49, help="Number of generated frames.")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Denoising steps per rollout.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale.")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, seeds everything and re-uses the same initial noise at every step "
                             "(the single-prompt tuning setting). Leave unset for regular training.")
    # ---- reward -----------------------------------------------------------------------
    parser.add_argument("--reward_fn", type=str, default="QwenReward",
                        choices=["QwenReward", "InternVLReward", "PickScoreReward"])
    parser.add_argument("--vlm_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Local path or Hugging Face id of the frozen VLM critic.")
    parser.add_argument("--reward_fn_kwargs", type=str, default=None,
                        help="JSON dict of extra keyword arguments for non-VLM rewards.")
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Generated frames given to the VLM (N_f).")
    parser.add_argument("--ref_frames_num_phy", type=int, default=10,
                        help="Reference frames appended for physics queries.")
    parser.add_argument("--max_frame_pixels", type=int, default=65536,
                        help="Maximum pixels per frame at the VLM input.")
    parser.add_argument("--mix_loss", action="store_true",
                        help="Score the fixed visual-quality (VQ) query in addition to each sampled query.")
    parser.add_argument("--binary_answer", action="store_true",
                        help="Ablation: supervise only the first 10 answer tokens (decision without explanation).")
    parser.add_argument("--yes_token_only", action="store_true",
                        help="Supervise only the `Yes` token (Vanilla-DRF style scalar reward).")
    # ---- back-propagation through the sampler -------------------------------------------
    parser.add_argument("--backprop_strategy", choices=["last", "tail", "uniform", "random"], default="tail")
    parser.add_argument("--backprop_num_steps", type=int, default=3,
                        help="Number of denoising steps that keep gradients (tail/uniform/random).")
    parser.add_argument("--backprop_step_list", nargs="+", type=int, default=None,
                        help="Explicit list of steps that keep gradients; overrides --backprop_strategy.")
    parser.add_argument("--backprop_random_start_step", type=int, default=0)
    parser.add_argument("--backprop_random_end_step", type=int, default=24)
    parser.add_argument("--stop_latent_model_input_gradient", action="store_true",
                        help="Detach the latent fed to the transformer at every step.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Activation checkpointing inside the transformer (needed to fit in memory).")
    # ---- optimisation -------------------------------------------------------------------
    parser.add_argument("--train_batch_size", type=int, default=1, help="Only 1 is supported.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=3e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-10)
    parser.add_argument("--max_grad_norm", type=float, default=50.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--low_vram", action="store_true",
                        help="Move the text encoder to the CPU after encoding each prompt.")
    # ---- logging / checkpoints ----------------------------------------------------------
    parser.add_argument("--output_dir", type=str, default="output/diffusion_drf")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="diffusion-drf")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='A checkpoint directory name inside --output_dir, or "latest".')
    parser.add_argument("--save_video_steps", type=int, default=10,
                        help="Save the generated training sample and the VLM predictions every N steps.")
    args = parser.parse_args()

    if args.train_batch_size != 1:
        raise ValueError("Only --train_batch_size=1 is supported (use more GPUs or gradient accumulation).")
    if args.reward_decoder != "wan" and args.reward_decoder_path is None:
        raise ValueError("--reward_decoder_path is required when --reward_decoder is not `wan`.")
    if args.backprop_step_list is not None:
        assert all(0 <= step <= args.num_inference_steps - 1 for step in args.backprop_step_list)
    else:
        if args.backprop_strategy in ("tail", "uniform", "random"):
            assert args.backprop_num_steps <= args.num_inference_steps - 1
        if args.backprop_strategy == "random":
            assert args.backprop_random_start_step <= args.backprop_random_end_step <= args.num_inference_steps - 1
    return args


def load_reward_decoder(args, config):
    """VAE whose decoder maps latents to pixels inside the differentiable reward path."""
    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    if args.reward_decoder == "wan":
        vae_path = os.path.join(args.pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae"))
    else:  # lightvae: same architecture and latent space as the Wan2.1 VAE with 1/4 of the channels
        vae_path = args.reward_decoder_path
        vae_kwargs["base_dim"] = 24
    return AutoencoderKLWan.from_pretrained(vae_path, additional_kwargs=vae_kwargs)


def select_backprop_steps(args):
    if args.backprop_step_list is not None:
        return args.backprop_step_list
    if args.backprop_strategy == "last":
        return [args.num_inference_steps - 1]
    if args.backprop_strategy == "tail":
        return list(range(args.num_inference_steps))[-args.backprop_num_steps:]
    if args.backprop_strategy == "uniform":
        interval = args.num_inference_steps // args.backprop_num_steps
        random_start = random.randint(0, interval)
        return [random_start + i * interval for i in range(args.backprop_num_steps)]
    if args.backprop_strategy == "random":
        return random.sample(
            range(args.backprop_random_start_step, args.backprop_random_end_step + 1), args.backprop_num_steps
        )
    raise ValueError(f"Invalid backprop strategy: {args.backprop_strategy}.")


def resize_reference(ref_frames, height, width, device, dtype):
    """uint8 ``[n, H, W, C]`` reference frames -> float ``[n, C, height, width]`` in [0, 1]."""
    ref_frames = ref_frames.to(device, dtype).permute(0, 3, 1, 2)
    ref_frames = transforms.functional.resize(
        ref_frames, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=False,
    ).float()
    return ref_frames / 255.0


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    config = OmegaConf.load(args.config_path)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)
    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16

    # ---- models -----------------------------------------------------------------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    vae = load_reward_decoder(args, config)
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    )

    # Only the diffusion transformer is trained; VAE, text encoder and critic stay frozen.
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(True)

    if args.transformer_path is not None:
        logger.info(f"Loading transformer weights from {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        missing, unexpected = transformer3d.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # Checkpoints store the transformer in diffusers format next to the optimizer state.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = WanTransformer3DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_params = [p for p in transformer3d.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---- reward -----------------------------------------------------------------------
    is_vlm_reward = args.reward_fn in VLM_REWARDS
    if is_vlm_reward:
        reward_fn_kwargs = dict(
            vlm_path=args.vlm_path,
            lambda_main=args.loss_weight,
            mix_loss=args.mix_loss,
            num_frames=args.num_frames,
            binary_answer=args.binary_answer,
            yes_token_only=args.yes_token_only,
        )
    else:
        reward_fn_kwargs = json.loads(args.reward_fn_kwargs) if args.reward_fn_kwargs else {}
    loss_fn = getattr(reward_fn, args.reward_fn)(device=accelerator.device, dtype=weight_dtype, **reward_fn_kwargs)

    # ---- data -------------------------------------------------------------------------
    # One row per query. Every process draws its own random order.
    data = pd.read_csv(args.data_path, sep="\t", encoding="utf-8-sig")
    data = data.sample(frac=1).reset_index(drop=True)

    num_update_steps_per_epoch = math.ceil(len(data) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    transformer3d, optimizer, lr_scheduler = accelerator.prepare(transformer3d, optimizer, lr_scheduler)

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if k != "backprop_step_list"}
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num queries = {len(data)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    height = int(args.train_sample_height // 16 * 16)
    width = int(args.train_sample_width // 16 * 16)
    do_classifier_free_guidance = args.guidance_scale > 1.0
    vae_scale_factor = vae.config.spatial_compression_ratio
    latent_shape = [
        args.train_batch_size,
        vae.config.latent_channels,
        int((args.video_length - 1) // vae.config.temporal_compression_ratio + 1) if args.video_length != 1 else 1,
        height // vae_scale_factor,
        width // vae_scale_factor,
    ]
    patch_size = accelerator.unwrap_model(transformer3d).config.patch_size
    seq_len = math.ceil(latent_shape[3] * latent_shape[4] / (patch_size[1] * patch_size[2]) * latent_shape[2])
    sample_dir = os.path.join(args.output_dir, "train_samples")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_reward = 0.0
        for idx in range(num_update_steps_per_epoch):
            train_batch = data.iloc[idx]
            train_prompt = [train_batch["prompt"]]
            video_ids = [train_batch["video_id"]]
            questions = list(ast.literal_eval(train_batch["questions"])) if is_vlm_reward else None
            gt_answers = list(ast.literal_eval(train_batch["gt_answers"])) if is_vlm_reward else None
            ref_video_path = os.path.join(args.video_root, train_batch["real_video_path"]) if is_vlm_reward else None
            logger.info(f"train_prompt: {train_prompt}")

            # ---- text conditioning ------------------------------------------------------
            if args.low_vram:
                torch.cuda.empty_cache()
                text_encoder.to(accelerator.device)
            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds = encode_prompt(
                    tokenizer,
                    text_encoder,
                    train_prompt,
                    negative_prompt=[NEGATIVE_PROMPT] * len(train_prompt),
                    device=accelerator.device,
                    dtype=weight_dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            if do_classifier_free_guidance:
                prompt_embeds = negative_prompt_embeds + prompt_embeds
            if args.low_vram:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            if getattr(noise_scheduler.config, "use_dynamic_shifting", False):
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device, mu=1)
            else:
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
            timesteps = noise_scheduler.timesteps

            with accelerator.accumulate(transformer3d):
                # ---- 1. rollout with truncated back-propagation ---------------------------
                if args.seed:
                    noise_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    latents = torch.randn(*latent_shape, generator=noise_generator, device=accelerator.device, dtype=weight_dtype)
                    step_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                else:
                    latents = torch.randn(*latent_shape, device=accelerator.device, dtype=weight_dtype)
                    step_generator = None
                if hasattr(noise_scheduler, "init_noise_sigma"):
                    latents = latents * noise_scheduler.init_noise_sigma
                extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, step_generator, args.eta)

                backprop_step_list = select_backprop_steps(args)
                for i, t in enumerate(tqdm(timesteps, disable=not accelerator.is_local_main_process, leave=False)):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if hasattr(noise_scheduler, "scale_model_input"):
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    t_expand = torch.tensor([t] * latent_model_input.shape[0], device=accelerator.device).to(
                        dtype=latent_model_input.dtype
                    )
                    if args.stop_latent_model_input_gradient:
                        latent_model_input = latent_model_input.detach()

                    with torch.autocast("cuda", dtype=weight_dtype):
                        noise_pred = transformer3d(
                            x=latent_model_input,
                            context=prompt_embeds,
                            t=t_expand,
                            seq_len=seq_len,
                        )
                    # Steps outside the list are treated as constants (stop-gradient).
                    if i not in backprop_step_list:
                        noise_pred = noise_pred.detach()

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # ---- 2. decode and sub-sample frames ---------------------------------------
                vae_device = next(vae.parameters()).device
                vae_dtype = next(vae.parameters()).dtype
                video = vae.decode(latents.to(vae_device, vae_dtype))[0]  # [B, 3, T, H, W] in [-1, 1]
                indices_to_keep = torch.linspace(0, video.shape[2] - 1, args.num_frames).round_().long()
                frames = video[:, :, indices_to_keep].permute(0, 2, 1, 3, 4)  # [B, N_f, 3, H, W]

                resized_height, resized_width = smart_resize(
                    frames.shape[-2], frames.shape[-1], factor=28, min_pixels=16384, max_pixels=args.max_frame_pixels,
                )
                frames_resized = torch.stack([
                    transforms.functional.resize(
                        clip, [resized_height, resized_width], interpolation=InterpolationMode.BILINEAR, antialias=False,
                    ).float()
                    for clip in frames
                ])
                frames_resized = (frames_resized.clamp(-1, 1) / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]

                # ---- 3. assemble the VLM inputs --------------------------------------------
                if is_vlm_reward:
                    # Physics (Phy) queries compare the generated clip with `ref_frames_num_phy`
                    # reference frames; all other queries use one reference frame (for VQ).
                    ref_frames_num = 1
                    is_phy_query = "physics-related defects" in questions[0]
                    if is_phy_query:
                        ref_frames_num = args.ref_frames_num_phy
                        # The annotation text was written for 24 generated + 12 reference frames.
                        questions[0] = questions[0].replace("(the last 12 frames)", f"(the last {ref_frames_num} frames)")
                        questions[0] = questions[0].replace("(the first 24 frames)", f"(the first {args.num_frames} frames)")
                        ref_frames = resize_reference(
                            extract_ref_frames(ref_video_path, ref_frames_num), resized_height, resized_width, vae_device, vae_dtype
                        )
                        frames_resized = torch.cat((frames_resized, ref_frames.unsqueeze(0)), dim=1)

                    if args.mix_loss:
                        # Second clip for the VQ query: generated frames followed by reference frames.
                        ref_frames = resize_reference(
                            extract_ref_frames(ref_video_path, ref_frames_num), resized_height, resized_width, vae_device, vae_dtype
                        )
                        num_generated = args.num_frames - 1 if ref_frames_num == 1 else args.num_frames
                        frames_vq = torch.cat((frames_resized[:, :num_generated], ref_frames.unsqueeze(0)), dim=1)
                        reward_inputs = [frames_resized.squeeze(0), frames_vq.squeeze(0)]
                    else:
                        reward_inputs = frames_resized

                    # ---- 4. differentiable reward ------------------------------------------
                    loss, reward, pred_answer = loss_fn(reward_inputs, train_prompt, [[questions, gt_answers]])
                else:
                    loss, reward = loss_fn(frames_resized, train_prompt)
                    pred_answer = None

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_reward = accelerator.gather(reward.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_reward += avg_reward.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if global_step % args.save_video_steps == 0:
                    os.makedirs(sample_dir, exist_ok=True)
                    stem = os.path.join(sample_dir, f"sample-{global_step}-{accelerator.process_index}")
                    video_vis = (video.detach().clamp(-1, 1) / 2 + 0.5).clamp(0, 1)
                    save_videos_grid(video_vis.to(torch.float32).cpu(), stem + ".mp4", fps=8)
                    if is_vlm_reward:
                        with open(stem + ".json", "w") as f:
                            json.dump({
                                "prompt": train_prompt, "video_ids": video_ids, "question": questions,
                                "gt_answer": gt_answers, "pred_answer": pred_answer,
                            }, f, indent=4)

                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    accelerator.log({"total_norm": total_norm}, step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_reward": train_reward}, step=global_step)
                train_loss = 0.0
                train_reward = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for removing_checkpoint in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                    gc.collect()
                    torch.cuda.empty_cache()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(
                step_loss=loss.detach().item(), step_reward=reward.mean().detach().item(), lr=lr_scheduler.get_last_lr()[0]
            )
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
