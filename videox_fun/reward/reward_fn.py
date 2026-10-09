"""VLM-based differentiable rewards for Diffusion-DRF, plus the PickScore baseline.

The reward of Diffusion-DRF is the teacher-forced next-token likelihood that a frozen
vision-language model (VLM) assigns to a reference answer, given the generated video and
a structured question. Gradients flow from that likelihood through the VLM vision tower
into the video frames (and from there into the diffusion model).

All rewards share the call signature

    loss, reward, extra = reward_fn(batch_frames, batch_prompt, batch_questions)

where ``batch_frames`` holds video frames in ``[0, 1]`` (see each class for the layout).
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange


# --------------------------------------------------------------------------------------
# Visual-quality (VQ) query
# --------------------------------------------------------------------------------------
# The VQ query is the same for every training sample, so it is built here instead of being
# stored in the annotation file. The wording (including the doubled quotes, an artefact of
# how the prompt was first stored) is kept fixed, because the reference answer below is
# tied to it.
def build_vq_question(num_frames: int, num_ref_frames: int) -> str:
    return f'''Compare the test video (the first {num_frames} frames) to the reference frame (the last {num_ref_frames} frame) and decide if the video shows obvious visual-quality (VQ) defects relative to the reference.\nConsider only: blur (defocus/motion), compression artifacts (blocking/ringing/mosquito), noise/grain, banding, flicker, rolling-shutter, aliasing/moire, over-smoothing and detail loss vs reference (missing fine textures/micro-contrast).\nReturn ONLY this JSON (no timestamps):\n{{\n  ""has_obvious_defect"": Yes | No,\n  ""dominant_issue"": ""none|defocus_blur|motion_blur|blocking|ringing|mosquito_noise|grain_noise|banding|flicker|rolling_shutter|aliasing|moire|over_smoothing|detail_loss"",\n  ""detail_loss_vs_reference"": Yes | No,\n  ""evidence"": [""short visual cues vs reference, e.g., 'softer edges than reference', 'block edges visible around text', 'fine fabric texture missing vs reference'""],\n}}\nRules:\n- Compare against the reference frames’ look (sharpness, texture, edges, tones).'''


VQ_REFERENCE_ANSWER = '{\n  ""has_obvious_defect"": No,\n  ""dominant_issue"": ""none"",\n ""detail_loss_vs_reference"": ""No"",\n ""evidence"": []\n}'
INTERNVL_VQ_REFERENCE_ANSWER = '{\n  ""has_obvious_defect"": No,\n  ""dominant_issue"": ""none"",\n  ""detail_loss_vs_reference"": ""No"",\n  ""evidence"": []\n}'


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def _answer_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    yes_token_id: Optional[int] = None,
    yes_token_only: bool = False,
) -> torch.Tensor:
    """Mean next-token cross-entropy over the supervised (label != -100) positions.

    This is the valid-token mean of the paper's objective. With ``yes_token_only`` the loss
    is restricted to the positions whose target is the ``Yes`` token (falling back to all
    supervised tokens if a sample contains none); this is the answer-only variant.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if yes_token_only and yes_token_id is not None:
        yes_mask = shift_labels == yes_token_id
        if yes_mask.any():
            return F.cross_entropy(shift_logits[yes_mask], shift_labels[yes_mask])
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
    )


# --------------------------------------------------------------------------------------
# Qwen2-VL / Qwen2.5-VL critic (default)
# --------------------------------------------------------------------------------------
class QwenReward:
    """Teacher-forced answer likelihood under a frozen Qwen2-VL / Qwen2.5-VL model.

    Args:
        vlm_path: local path or Hugging Face id, e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``.
        lambda_main: scalar multiplier applied to the loss.
        mix_loss: additionally score the fixed visual-quality (VQ) query on a second clip
            (generated frames followed by reference frames).
        num_frames: number of generated frames given to the VLM (``N_f`` in the paper).
        binary_answer: supervise only the first 10 answer tokens (the Yes/No decision
            without the explanation). Used for the "binary question" ablation.
        yes_token_only: supervise only the ``Yes`` token. Used for Vanilla-DRF.

    Call arguments:
        batch_frames: without ``mix_loss`` a tensor ``[1, T, C, H, W]``; with ``mix_loss``
            a list ``[clip, vq_clip]`` of two ``[T, C, H, W]`` tensors. Values in ``[0, 1]``,
            already resized to a multiple of the VLM patch size.
        batch_questions: ``[[questions, reference_answers]]`` (batch size 1), two lists of
            equal length.

    Returns:
        ``(loss, reward, predictions)`` where ``reward = -loss`` (detached) is the mean
        log-likelihood of the reference tokens and ``predictions`` are the greedy
        teacher-forced decodings, for logging.
    """

    def __init__(
        self,
        vlm_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device="cpu",
        dtype=torch.bfloat16,
        lambda_main: float = 1.0,
        mix_loss: bool = False,
        num_frames: int = 10,
        binary_answer: bool = False,
        yes_token_only: bool = False,
    ):
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration,
                                  Qwen2VLForConditionalGeneration)

        self.lambda_main = lambda_main
        self.mix_loss = mix_loss
        self.num_frames = num_frames
        self.binary_answer = binary_answer
        self.yes_token_only = yes_token_only

        print(f"Loading VLM critic from {vlm_path} ...")
        if "Qwen2.5-VL" in vlm_path:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "Qwen2-VL" in vlm_path:
            model_cls = Qwen2VLForConditionalGeneration
        else:
            raise ValueError(
                f"Cannot infer the model class from `{vlm_path}`; the path must contain "
                "'Qwen2.5-VL' or 'Qwen2-VL'."
            )
        self.model = model_cls.from_pretrained(vlm_path, torch_dtype="auto", device_map=None)
        self.processor = AutoProcessor.from_pretrained(vlm_path)

        self.model.to(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        # The critic is frozen, but gradients must flow through it to the video, so
        # activation checkpointing is what keeps its memory footprint manageable.
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        tokenizer = self.processor.tokenizer
        self.assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]

    def _chat_text(self, question: str, answer: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "<video>"},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Supervise the assistant answer (through `<|im_end|>` and the newline after it)."""
        labels = torch.full_like(input_ids, -100)
        for i in range(len(labels)):
            ids = input_ids[i].tolist()
            length = len(ids)
            pos = 0
            while pos < length:
                if ids[pos] == self.assistant_token_id:
                    ans_start = pos + 2  # skip "assistant" and the newline that follows it
                    ans_end = ans_start
                    while ans_end < length and ids[ans_end] != self.im_end_token_id:
                        ans_end += 1
                    if ans_end < length:
                        if self.binary_answer:
                            ans_len = min(10, ans_end + 2 - ans_start)
                            labels[i, ans_start:ans_start + ans_len] = input_ids[i, ans_start:ans_start + ans_len]
                        else:
                            labels[i, ans_start:ans_end + 2] = input_ids[i, ans_start:ans_end + 2]
                        pos = ans_end
                pos += 1
        return labels

    def __call__(
        self,
        batch_frames,
        batch_prompt: Optional[List[str]] = None,
        batch_questions: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        questions, answers = batch_questions[0][0], batch_questions[0][1]
        texts, video_batch = [], []
        for question, answer in zip(questions, answers):
            texts.append(self._chat_text(question, answer))
            video_batch.append(batch_frames[0])
            if self.mix_loss:
                # NOTE: the frame counts quoted in the VQ question are derived from the
                # length of the *question* clip.
                clip_len = len(batch_frames[0])
                vq_question = build_vq_question(self.num_frames, clip_len - self.num_frames)
                texts.append(self._chat_text(vq_question, VQ_REFERENCE_ANSWER))
                video_batch.append(batch_frames[1])

        # Frames are already in [0, 1] and at the target resolution; the processor must
        # neither rescale, resize nor resample them, or the gradient path would be lost.
        inputs = self.processor(
            text=texts,
            images=None,
            videos=video_batch,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
            do_sample_frames=False,
        ).to(self.model.device)

        labels = self._build_labels(inputs["input_ids"])
        logits = self.model.forward(**inputs).logits
        labels = labels.to(logits.device)

        loss = self.lambda_main * _answer_token_loss(
            logits, labels, yes_token_id=self.yes_token_id, yes_token_only=self.yes_token_only
        )

        pred_sentences = []
        for i in range(len(labels)):
            idx = torch.nonzero(labels[i] != -100).squeeze(-1)
            pred_sentences.append(self.processor.decode(logits[i, idx].argmax(-1)))
        return loss, -loss.detach(), pred_sentences


# --------------------------------------------------------------------------------------
# InternVL3 critic
# --------------------------------------------------------------------------------------
class InternVLReward:
    """Teacher-forced answer likelihood under a frozen InternVL3 model.

    Same interface as :class:`QwenReward`. Every frame is resized to the model's native
    square resolution and at most ``num_frames`` frames of each clip are used.
    """

    def __init__(
        self,
        vlm_path: str = "OpenGVLab/InternVL3-1B",
        device="cpu",
        dtype=torch.bfloat16,
        lambda_main: float = 1.0,
        mix_loss: bool = False,
        num_frames: int = 10,
        binary_answer: bool = False,
        yes_token_only: bool = False,
    ):
        from transformers import AutoModel, AutoTokenizer

        if yes_token_only:
            raise NotImplementedError("`yes_token_only` is only implemented for QwenReward.")
        self.lambda_main = lambda_main
        self.mix_loss = mix_loss
        self.num_frames = num_frames
        self.binary_answer = binary_answer

        print(f"Loading VLM critic from {vlm_path} ...")
        self.model = AutoModel.from_pretrained(
            vlm_path, torch_dtype="auto", device_map=None, trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_context_token = "<IMG_CONTEXT>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.img_context_token)
        self.model.img_context_token_id = self.img_context_token_id
        self.image_size = int(
            getattr(self.model.config, "force_image_size", None)
            or getattr(self.model.config.vision_config, "image_size", 448)
        )
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def _prepare_video_pixels(self, video: torch.Tensor) -> Tuple[torch.Tensor, int]:
        assert video.dim() == 4, f"Expected a 4D video tensor, got {video.shape}"
        if video.shape[0] == 3:
            video_tchw = video.permute(1, 0, 2, 3)
        elif video.shape[1] == 3:
            video_tchw = video
        else:
            raise AssertionError(f"Expected [C, T, H, W] or [T, C, H, W] with C=3, got {video.shape}")

        n_frames = min(int(video_tchw.shape[0]), int(self.num_frames))
        video_tchw = video_tchw[:n_frames]
        if torch.min(video_tchw).item() < 0:
            video_tchw = (video_tchw + 1.0) / 2.0
        video_tchw = video_tchw.clamp(0.0, 1.0)
        video_tchw = F.interpolate(
            video_tchw, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False,
        )
        mean = self.imagenet_mean.to(device=video_tchw.device, dtype=video_tchw.dtype).view(1, 3, 1, 1)
        std = self.imagenet_std.to(device=video_tchw.device, dtype=video_tchw.dtype).view(1, 3, 1, 1)
        return (video_tchw - mean) / std, n_frames

    def _build_text_pair(self, question: str, answer: str, num_patches: int) -> Tuple[str, str]:
        user_text = f"<image>\n{question}"
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prefix_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}],
                    tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                prefix_text = f"User: {user_text}\nAssistant:"
                full_text = f"{prefix_text} {answer}"
        else:
            prefix_text = f"User: {user_text}\nAssistant:"
            full_text = f"{prefix_text} {answer}"

        img_tokens = (
            self.img_start_token
            + self.img_context_token * (int(self.model.num_image_token) * int(num_patches))
            + self.img_end_token
        )
        if "<image>" in prefix_text:
            prefix_text = prefix_text.replace("<image>", img_tokens, 1)
        else:
            prefix_text = img_tokens + "\n" + prefix_text
        if "<image>" in full_text:
            full_text = full_text.replace("<image>", img_tokens, 1)
        else:
            full_text = img_tokens + "\n" + full_text
        return full_text, prefix_text

    def __call__(
        self,
        batch_frames,
        batch_prompt: Optional[List[str]] = None,
        batch_questions: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        questions, answers = batch_questions[0][0], batch_questions[0][1]
        text_full, text_prefix, pixel_values_chunks = [], [], []
        for question, answer in zip(questions, answers):
            video_pixels, num_patches = self._prepare_video_pixels(batch_frames[0])
            full_t, prefix_t = self._build_text_pair(question, answer, num_patches)
            text_full.append(full_t)
            text_prefix.append(prefix_t)
            pixel_values_chunks.append(video_pixels)
            if self.mix_loss:
                vq_clip = batch_frames[1] if len(batch_frames) > 1 else batch_frames[0]
                video_pixels_vq, num_patches_vq = self._prepare_video_pixels(vq_clip)
                # The VQ clip is truncated to `num_frames` frames, the last of which is the
                # reference frame. The InternVL runs used a plain apostrophe in the question
                # and a slightly different indentation in the reference answer.
                vq_question = build_vq_question(self.num_frames, 1).replace("\u2019", "'")
                full_vq, prefix_vq = self._build_text_pair(
                    vq_question, INTERNVL_VQ_REFERENCE_ANSWER, num_patches_vq
                )
                text_full.append(full_vq)
                text_prefix.append(prefix_vq)
                pixel_values_chunks.append(video_pixels_vq)

        device = next(self.model.parameters()).device
        input_batch = self.tokenizer(text_full, return_tensors="pt", padding=True)
        prefix_lens = [len(self.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in text_prefix]
        input_ids = input_batch["input_ids"].to(device)
        attention_mask = input_batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for i, plen in enumerate(prefix_lens):
            labels[i, :min(plen, labels.shape[1])] = -100
            if self.binary_answer:
                ans_idx = torch.nonzero(labels[i] != -100).squeeze(-1)
                if ans_idx.numel() > 10:
                    labels[i, ans_idx[10:]] = -100

        pixel_values = torch.cat(pixel_values_chunks, dim=0).to(device=device, dtype=self.model.dtype)
        vit_embeds = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        bsz, seq_len, channels = input_embeds.shape
        input_embeds_flat = input_embeds.view(bsz * seq_len, channels)
        selected = input_ids.view(bsz * seq_len) == self.img_context_token_id
        vit_flat = vit_embeds.reshape(-1, channels).to(input_embeds_flat.dtype)
        if selected.sum().item() != vit_flat.shape[0]:
            raise RuntimeError(
                f"InternVL token/image mismatch: {selected.sum().item()} image tokens vs "
                f"{vit_flat.shape[0]} visual features. Check the <image> placeholder and frame count."
            )
        input_embeds_flat[selected] = vit_flat
        input_embeds = input_embeds_flat.view(bsz, seq_len, channels)

        logits = self.model.language_model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True,
        ).logits
        loss = self.lambda_main * _answer_token_loss(logits, labels)

        pred_sentences = []
        for i in range(labels.shape[0]):
            idx = torch.nonzero(labels[i] != -100).squeeze(-1)
            if idx.numel() == 0:
                pred_sentences.append("")
                continue
            pred_sentences.append(self.tokenizer.decode(logits[i, idx].argmax(-1), skip_special_tokens=True))
        return loss, -loss.detach(), pred_sentences


# --------------------------------------------------------------------------------------
# PickScore baseline (scalar reward)
# --------------------------------------------------------------------------------------
class PickScoreReward:
    """[PickScore](https://github.com/yuvalkirstain/PickScore) frame-wise reward (baseline).

    ``batch_frames`` is ``[B, T, C, H, W]`` in ``[0, 1]``; returns ``(loss, reward)``.
    """

    def __init__(
        self,
        model_path: str = "yuvalkirstain/PickScore_v1",
        device="cpu",
        dtype=torch.float16,
        max_reward: Optional[float] = 1,
        loss_scale: float = 1,
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale
        # https://huggingface.co/yuvalkirstain/PickScore_v1/blob/main/preprocessor_config.json
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=self.dtype)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.dtype).eval().to(device)
        self.model.requires_grad_(False)

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: List[str], batch_questions=None):
        assert batch_frames.shape[0] == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b t c h w -> t b c h w")
        batch_loss, batch_reward = 0, 0
        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            text_inputs = self.processor(
                text=batch_prompt, padding=True, truncation=True, max_length=77, return_tensors="pt",
            ).to(self.device)
            image_features = self.model.get_image_features(pixel_values=image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
            text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)

            reward = torch.diagonal(image_features @ text_features.T)
            if self.max_reward is None:
                loss = (-1 * reward) * self.loss_scale
            else:
                loss = abs(reward - self.max_reward) * self.loss_scale
            batch_loss, batch_reward = batch_loss + loss.mean(), batch_reward + reward.mean()

        return batch_loss / batch_frames.shape[0], batch_reward / batch_frames.shape[0]
