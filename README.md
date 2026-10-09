# Diffusion-DRF: Free, Rich, and Differentiable Reward for Video Diffusion Fine-Tuning

**Unofficial, third-party reproduction** of **Diffusion-DRF** (NeurIPS 2026).
[Paper](https://arxiv.org/abs/2601.04153) | [Official project page](https://snap-research.github.io/diffusion-drf/)

> **Note.** This repository is a third-party reproduction and **not the official release**.
> We do not have access to the H3AE decoder used in the paper, so the public LightVAE is
> borrowed as a substitute in the reward path (see [Weights](#weights)). Results obtained with
> this code may differ from the numbers reported in the paper.

Diffusion-DRF (Differentiable Reward Flow) fine-tunes a text-to-video diffusion model with
feedback from a **frozen, off-the-shelf vision-language model (VLM)**. No reward model is
trained and no human preference data is collected. Each training caption is decomposed into
structured Yes/No questions with free-form explanations along three dimensions,
text-video alignment (TA), physical fidelity (Phy) and visual quality (VQ). The reward is the
teacher-forced likelihood the VLM assigns to the reference answers given the *generated*
video, and it is back-propagated through the VLM, the video decoder and the last denoising
steps into the diffusion transformer.

This repository contains

- the training code for Wan2.1-T2V (`scripts/train_drf.py`),
- the VLM rewards (`videox_fun/reward/reward_fn.py`) for Qwen2.5-VL and InternVL3 critics,
- the automatic annotation pipeline (`annotation/`),
- the sampling script used for evaluation (`inference/predict_t2v.py`).

## Installation

```bash
git clone https://github.com/YFWang1999/Diffusion-DRF.git && cd Diffusion-DRF
# install the PyTorch build matching your CUDA version first
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional, speeds up the VLM
```

The code was developed with `transformers==4.57`; the differentiable video path through the
Qwen2.5-VL processor depends on that version's video processor.

## Weights

| What | Source | Used as |
|---|---|---|
| Wan2.1-T2V-1.3B | [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) (original layout) | generator, `--pretrained_model_name_or_path` |
| Qwen2.5-VL-7B-Instruct | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | frozen critic, `--vlm_path` |
| LightVAE (Wan2.1) | [`lightx2v/Autoencoders`](https://huggingface.co/lightx2v/Autoencoders), `lightvaew2_1.safetensors` | reward-path decoder, `--reward_decoder_path` |

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan2.1-T2V-1.3B
huggingface-cli download lightx2v/Autoencoders lightvaew2_1.safetensors --local-dir models
```

**Reward-path decoder.** Decoding 49 frames with gradients through the official Wan2.1 VAE is
very memory hungry. The paper uses a lightweight H3AE decoder in the reward path. We do not
have access to H3AE, so this reproduction borrows the public LightVAE as a substitute, a
channel-pruned Wan2.1 VAE that shares its latent space and loads into the same model class. On a 512x288, 49-frame clip, decoding plus
back-propagation takes about 14.5 GB with LightVAE, while the official VAE does not fit on a
48 GB GPU. Pass `--reward_decoder wan` to use the official VAE. Sampling at inference time
always uses the official VAE.

## Data

Training needs captioned real videos and an annotation file with one structured query per
row. The format is described in
[`data/README.md`](data/README.md). To annotate your own captioned videos, see
[Annotation pipeline](#annotation-pipeline).

## Training

```bash
MODEL_DIR=models/Wan2.1-T2V-1.3B \
VLM_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
DECODER_PATH=models/lightvaew2_1.safetensors \
DATA_PATH=data/drf_train_annotations.tsv \
VIDEO_ROOT=data/videos \
bash scripts/train_drf.sh
```

`scripts/train_drf.sh` follows the main setting described in the paper: 512x288, 49 frames, 25 denoising
steps with classifier-free guidance 6.0, `N_f = 10` generated frames and 10 reference frames
for the critic, gradients through the last 3 denoising steps, AdamW with learning rate 1e-5,
one sample per GPU on 16 GPUs. Checkpoints are written every 500 steps to
`<output_dir>/checkpoint-<step>/transformer/`; the paper reports the checkpoint at step 1500.

**Memory.** The paper's setting was run on 80 GB GPUs. To fit smaller GPUs, in order of
effect: use a smaller critic (`--vlm_path Qwen/Qwen2.5-VL-3B-Instruct`), lower
`--max_frame_pixels` (e.g. 16384), reduce `--num_frames` / `--ref_frames_num_phy`, reduce
`--backprop_num_steps`, and add `--low_vram`. Fewer GPUs can be compensated with
`--gradient_accumulation_steps`.

### Useful options

| Option | Effect |
|---|---|
| `--mix_loss` | adds the reference-guided VQ query to every sampled TA/Phy query (on in the main setting) |
| `--num_frames`, `--ref_frames_num_phy` | generated frames `N_f` and reference frames given to the critic |
| `--backprop_num_steps` | number of final denoising steps that keep gradients |
| `--binary_answer` | supervise only the Yes/No decision, not the explanation (ablation) |
| `--yes_token_only` | supervise only the `Yes` token, i.e. a scalar logit-level reward as in Vanilla-DRF |
| `--reward_fn InternVLReward --vlm_path OpenGVLab/InternVL3-1B` | InternVL3 critic |
| `--reward_fn PickScoreReward` | scalar PickScore baseline (`scripts/train_pickscore.sh`) |
| `--seed N` | fixes the initial noise across steps, for single-prompt tuning on a one-video annotation file |

## Inference

```bash
python inference/predict_t2v.py \
  --model_dir models/Wan2.1-T2V-1.3B \
  --transformer_path output/drf_wan1.3b_qwen7b/checkpoint-1500/transformer/diffusion_pytorch_model.safetensors \
  --prompt_file prompts.txt \
  --save_dir samples/drf
```

Omit `--transformer_path` to sample from the base model. The defaults are the sampling
settings of this reproduction: 512x288, 49 frames, 30 steps, guidance 6.0, shift 3 and
TeaCache with threshold 0.10. See [`evaluation/README.md`](evaluation/README.md) for the
benchmark protocol.

## Annotation pipeline

```bash
cd annotation
# 1. decompose captions into atomic scene elements with a text-only LLM
python decompose_captions.py --captions captions.tsv --output_json decompositions.json
# 2. build TA/Phy queries and answer the TA queries on the real videos with the VLM
python build_annotations.py --captions captions.tsv --decompositions decompositions.json \
  --video_root /path/to/videos --output ../data/my_annotations.tsv
```

`captions.tsv` is tab-separated with the columns `video_id`, `real_video_path`, `prompt`.
The question templates are in `annotation/prompts.py`. Both scripts accept `--num_shards/--shard_index` to
spread the work over several GPUs.

## Repository layout

```
scripts/train_drf.py          training loop (rollout, truncated back-propagation, reward)
scripts/train_drf.sh          main setting; scripts/train_pickscore.sh: PickScore baseline
videox_fun/reward/            VLM rewards (Qwen2.5-VL, InternVL3) and PickScore
videox_fun/{models,pipeline}  Wan2.1 transformer, VAE, text encoder and sampling pipeline
annotation/                   caption decomposition and query construction
inference/predict_t2v.py      sampling
config/wan2.1/                model configuration
```

## Citation

```bibtex
@inproceedings{wang2026diffusiondrf,
  title     = {Diffusion-DRF: Free, Rich, and Differentiable Reward for Video Diffusion Fine-Tuning},
  author    = {Wang, Yifan and Li, Yanyu and Qian, Gordon Guocheng and Tulyakov, Sergey and Fu, Yun and Kag, Anil},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements and license

This code builds on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache License 2.0);
the files under `videox_fun/` are adapted from it. It also relies on
[Wan2.1](https://github.com/Wan-Video/Wan2.1), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
[InternVL](https://github.com/OpenGVLab/InternVL), [LightX2V](https://github.com/ModelTC/LightX2V)
and [TeaCache](https://github.com/ali-vilab/TeaCache). The code is released under the
Apache License 2.0, see [`LICENSE`](LICENSE). Model weights and datasets are subject to their
own licenses.
