# Evaluation

The paper evaluates on [VBench-2.0](https://github.com/Vchitect/VBench) and on the
[VideoGen-Eval](https://github.com/AILab-CVC/VideoGen-Eval) prompt set. Videos for both are
sampled with `inference/predict_t2v.py` at its default settings (512x288, 49 frames, 30 steps,
guidance 6.0, shift 3, TeaCache threshold 0.10).

## VBench-2.0

1. Take the prompt lists of the VBench-2.0 repository. Sampling uses its prompt-augmented
   prompts for Wan (`prompts/prompt_aug/Wanx_full_text_aug.txt`), as recommended there for
   models trained on long captions.
2. Sample three videos per prompt with fresh noise, i.e. three runs of
   `inference/predict_t2v.py --seed -1` into three folders, for the base model and for each
   checkpoint. The Diversity dimension has its own prompt list and needs more samples per
   prompt; follow the VBench-2.0 instructions for it.
3. Copy each run to the VBench-2.0 naming scheme (`<prompt>-<index>.mp4`), using the
   original prompt list (same order as the augmented one) for the names:

   ```bash
   python evaluation/rename_for_vbench2.py --video_dir samples/drf_run0 \
     --prompt_file /path/to/VBench-2.0/prompts/VBench2_full_text.txt \
     --output_dir samples/drf_vbench2 --sample_index 0
   ```

4. Run the VBench-2.0 evaluation script on the renamed folder.

## VideoGen-Eval pairwise comparison

Sample the VideoGen-Eval prompts with a fixed seed (`--seed 42`) for the two models to be
compared, so that each pair starts from the same noise, and compare the pairs with
[VideoAlign](https://github.com/KwaiVGI/VideoAlign) or with human raters.
