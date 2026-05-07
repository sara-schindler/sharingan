# Sharingan — Agent Instructions

Multi-person gaze following research code (CVPR 2024). PyTorch Lightning + Hydra. See [README.md](README.md) for project overview, paper, demos, and download links for data/weights/checkpoints.

## Environment

- Python deps are pinned in [environment.yaml](environment.yaml) (Python 3.11, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Lightning 2.6.1, Hydra 1.3.2, transformers 4.44.0, wandb, boxmot). Create with `conda env create -f environment.yaml && conda activate sharingan`.
- No test suite, linter, formatter, or pre-commit config exists. Do **not** introduce one unless asked.
- License headers (SPDX `CC-BY-NC-4.0`) are present at the top of source files — preserve them when editing.

## Architecture map

- [main.py](main.py) — Hydra entry point. Builds `Experiment` from config and dispatches train/val/test/predict.
- [demo.py](demo.py) — standalone argparse script for image/video inference (uses YOLO head detector + boxmot tracker).
- [submit-experiment.sh](submit-experiment.sh) — preferred launcher: snapshots `src/` into `experiments/<date>/<time>/` then runs `main.py`. Edit the `--config-name` line inside to switch dataset.
- [src/config.py](src/config.py) — Hydra dataclass schema (`MyConfig`); fields are required (no defaults), values come from YAML.
- [src/conf/config.yaml](src/conf/config.yaml) — base config with `</path/to>/...` placeholders that must be replaced with real paths.
- [src/conf/config_gf.yaml](src/conf/config_gf.yaml), [config_vat.yaml](src/conf/config_vat.yaml), [config_cp.yaml](src/conf/config_cp.yaml) — dataset-specific overrides selected via `--config-name`.
- [src/experiments.py](src/experiments.py) — `Experiment` orchestrator: builds model, datamodule, callbacks (`ModelCheckpoint`, optional SWA), trainer; saves best to `./checkpoints/best.ckpt`.
- [src/modeling/sharingan.py](src/modeling/sharingan.py) — `SharinganModule` (LightningModule) wrapping `Sharingan` (MultiMAE ViT image tokens + ResNet18 gaze encoder fused via attention → heatmap decoder + in/out head).
- [src/datasets/](src/datasets/) — `gazefollow.py`, `videoattentiontarget.py`, `childplay.py`. Each exports a `*Dataset` and `*DataModule`. All require three roots: `root` (original images/anns), `root_heads` (precomputed head boxes from Zenodo), `root_annotations` (the new txt files in [data/](data/) for GazeFollow only).
- [src/losses.py](src/losses.py) — `compute_sharingan_loss` = weighted heatmap MSE + angular cosine + BCE; masked by `inout_gt`.
- [src/metrics.py](src/metrics.py) — torchmetrics: `PLAH`, `Distance`, `AUC`, `GFTestAUC`, `GFTestDistance`.
- [src/transforms.py](src/transforms.py), [src/visualize.py](src/visualize.py), [src/utils/common.py](src/utils/common.py) — augmentations, gaze drawing, geometry helpers (`generate_gaze_heatmap`, `spatial_argmax2d`, `square_bbox`, ...).
- [src/tracking.py](src/tracking.py) — W&B logger init and code-snapshot zip.

## Running things

- Train (GazeFollow): `python main.py --config-name config_gf` — or edit and run [submit-experiment.sh](submit-experiment.sh).
- Override anything via Hydra CLI, e.g.:
  - Test only: `python main.py experiment.task=test experiment.dataset=gazefollow test.checkpoint=checkpoints/gazefollow.pt`
  - Multi-task: `experiment.task=train+test`
- Demo: `python demo.py --input-dir <dir> --input-filename <video.mp4> --output-dir ./samples`.

## Project conventions & pitfalls

- **Paths must be configured before anything runs.** Every `</path/to>/...` in [src/conf/config.yaml](src/conf/config.yaml) needs real values, plus `model.pretraining.gaze360`, `model.pretraining.multivit`, and (for eval) `model.weights` / `test.checkpoint`. Do not invent paths — surface them to the user.
- **W&B entity is hard-coded** in [src/tracking.py](src/tracking.py) as `entity="wandb-username"`; users must replace this. Don't silently change it to a guess.
- **Required dirs** that aren't in git: `checkpoints/`, `weights/`, `experiments/`. Create on demand.
- **GazeFollow uses the new annotation files** in [data/](data/) (`train_annotations_new.txt`, `val_annotations_new.txt`). Test split uses original annotations to keep comparisons fair — don't replace them.
- **Hydra changes cwd** to the experiment output dir at runtime; treat relative paths in configs/code accordingly. The `submit-experiment.sh` flow expects to be launched from the repo root.
- **In/out BCE loss is disabled by default** (`loss.weight_bce=0`). Note when changing.
- **Demo head detector stays on YOLOv5 torch.hub.** The shipped head checkpoint in [weights/yolov5m_crowdhuman.pt](weights/yolov5m_crowdhuman.pt) is not forward-compatible with the newer `ultralytics.YOLO(...)` loader.
- **BoxMOT class names changed in newer releases.** In [demo.py](demo.py), imports use `DeepOcSort`/`ByteTrack`/`OcSort` names.
- **CUDA device selection in demo uses a runtime probe.** [demo.py](demo.py) now tests a real CUDA op before selecting GPU, avoiding false fallback from architecture-token checks on newer cards.
- Code style: type hints on signatures, snake_case functions, CamelCase classes, minimal docstrings. Match the surrounding style — don't add docstrings/comments to code you didn't change.
- Datasets support multi-person gaze with `num_people=2` (or `-1` for all). The model expects ≥1 head bbox per image; for GazeFollow the precomputed detections fill in extra people.

## When editing

- Changes to model/dataset/loss interfaces ripple through [src/experiments.py](src/experiments.py) and the YAML schema in [src/config.py](src/config.py) — update the dataclass and the YAML files together.
- Don't hand-edit files inside `experiments/<date>/<time>/` — those are run snapshots.
- For new Hydra fields, add them to both [src/config.py](src/config.py) (typed) and the relevant YAML, then reference via `cfg.<section>.<field>`.
