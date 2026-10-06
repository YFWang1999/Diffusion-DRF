# Evaluation

The paper evaluates on [VBench-2.0](https://github.com/Vchitect/VBench) and on the
[VideoGen-Eval](https://github.com/AILab-CVC/VideoGen-Eval) prompt set. Videos for both are
sampled with `inference/predict_t2v.py` at its default settings (512x288, 49 frames, 30 steps,
guidance 6.0, shift 3, TeaCache threshold 0.10).

## VBench-2.0

1. Take the prompt lists of the VBench-2.0 repository. We use its prompt-augmented prompts
   for Wan, as recommended there for models trained on long captions.
2. Sample the required number of videos per prompt with `--seed -1` (fresh noise per video),
   one run per sample index, for the base model and for each checkpoint.
3. Rename the videos to the file-name convention expected by VBench-2.0 and run its
   evaluation script on each dimension.

`predict_t2v.py` names videos by the 1-based line index of the prompt file
(`00000001.mp4`, ...), so the mapping back to prompts is the prompt file itself.

## VideoGen-Eval pairwise comparison

Sample the VideoGen-Eval prompts with a fixed seed (`--seed 42`) for the two models to be
compared, so that each pair starts from the same noise, and compare the pairs with
[VideoAlign](https://github.com/KwaiVGI/VideoAlign) or with human raters.
