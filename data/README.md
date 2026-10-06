# Training data

Diffusion-DRF is trained on captioned real videos. Each caption is used as the prompt for a
video sampled from the generator, and the paired real video serves only as the reference clip
of the physics (Phy) and visual-quality (VQ) queries. Real videos are never used as
reconstruction targets.

## Annotation file

A tab-separated file with one **query** per row:

| Column | Content |
|---|---|
| `video_id` | identifier of the real video |
| `real_video_path` | path of the real video, relative to `--video_root` |
| `prompt` | caption of the real video, used as the generation prompt |
| `questions` | Python list literal with one question string |
| `gt_answers` | Python list literal with the reference answer to that question |

A video appears in several rows: one per decomposed text-video alignment (TA) component and
one for the Phy query. The VQ query is identical for all samples, so it is not stored in the
file; `--mix_loss` adds it to every sampled row at training time.

- **TA** rows ask whether the generated video fulfils one component of the caption
  (environment, objects, motion, spatial layout, colours, lighting, text, camera motion). The
  reference answer is a Yes/No decision with a short explanation, produced by the VLM on the
  real video.
- **Phy** rows ask for physics-related defects with the real video as reference. Their
  question text mentions "the first 24 frames" and "the last 12 frames"; the training script
  rewrites these to the actual `--num_frames` and `--ref_frames_num_phy`.

## Annotations of the paper

The annotations used in the paper cover 5,077 captioned videos and 24,252 queries
(19,175 TA and 5,077 Phy). They will be made available on the project page. The videos
themselves come from public datasets and must be downloaded from their original sources and
placed under `--video_root` following the relative paths in `real_video_path`.

## Annotating your own videos

See the "Annotation pipeline" section of the top-level README.
