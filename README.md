<div align="center">

<img src="assets/argus_logo.png" width="200">

<h3>Argus: Metric Panoramic 3D Reconstruction for Indoor Scenes</h3>

<h4>ECCV 2026</h4>

<a href="https://argus-paper.realsee.ai" target="_blank"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2606.30047" target="_blank"><img src="https://img.shields.io/badge/arXiv-2606.30047-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/RealseeTechnology/argus-realsee3d" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="HuggingFace Model"></a>
<a href="https://huggingface.co/spaces/RealseeDeveloper/Argus" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="HuggingFace Demo"></a>
<a href="https://github.com/realsee-developer/RealSee3D" target="_blank"><img src="https://img.shields.io/badge/RealSee3D-Dataset-orange" alt="RealSee3D Dataset"></a>


<p><b><a href="https://www.realsee.ai">Realsee</a></b></p>
</div>

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/A511-git/Argus.git
cd Argus
pip install -r requirements.txt
pip install -e .
```

Download the pretrained weights:

```bash
# Authenticate with HuggingFace (required for gated model)
hf auth login

# Option 1: Auto-download via Python (cached by huggingface_hub)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='RealseeTechnology/argus-realsee3d', filename='argus_realsee3d.pt')"

# Option 2: Manual download
mkdir -p models
hf download RealseeTechnology/argus-realsee3d argus_realsee3d.pt --local-dir models
```

Run inference with a few lines of code:

```python
import torch
from huggingface_hub import hf_hub_download
from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360

# Download model weights (requires: hf auth login)
model_path = hf_hub_download(
    repo_id="RealseeTechnology/argus-realsee3d",
    filename="argus_realsee3d.pt",
)

# Load model
model = Argus(reorder_by_learning_ref=True, restore_metric_scale=True)
model.load_state_dict(torch.load(model_path)["model"], strict=False)
model.eval().cuda()

# Prepare input: panoramic images as tensor [S, 3, H, W], values in [0, 1]
images = ...  # your preprocessed ERP images

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    predictions = model(images.cuda())

# Extract camera extrinsics
extrinsic, conf = pose_encoding_to_extri360(pose_encoding=predictions["pose_enc"])

# Access depth and 3D points
depth = predictions["depth"]          # [B, S, H, W, 1]
depth_conf = predictions["depth_conf"]  # [B, S, H, W]
```

## 🚀 Standalone CLI Pipeline & 3D Gaussian Splatting

You can run automated, batch-capable 3D reconstruction and export photorealistic **3D Gaussian Splats** directly from the command line using `infer.py`:

```bash
# Basic reconstruction (auto-downloads weights from HuggingFace if omitted)
python infer.py --input path/to/scene_folder --output ./output

# Specify a local model checkpoint
python infer.py --input path/to/scene_folder --output ./output --model_path models/argus_realsee3d.pt

# High-resolution dense reconstruction (capped to 2K by default)
python infer.py --input path/to/scene_folder --output ./output --dense

# Generate photorealistic 3D Gaussian Splats (.ply) for SuperSplat / PlayCanvas / Luma
python infer.py --input path/to/scene_folder --output ./output --splat --ply_scale 1.2

# Outdoor scene reconstruction with extended depth cutoff (80m)
python infer.py --input path/to/outdoor_scene --output ./output --splat --no-ply_is_indoor

# Export interactive 3D GLB scene with color-coded camera frustums and open-top ceiling cut
python infer.py --input path/to/scene_folder --output ./output --export_glb --show_cam --ceiling_remove 25.0

# Batch process multiple scenes under a parent directory on Linux / Kaggle / Colab
python infer.py --input /path/to/scenes_parent_folder --output /path/to/output --dense --splat --model_path /path/to/argus_realsee3d.pt
```

### Generated Scene Outputs
For each scene (e.g. `Created_Scenes_pantry`), `infer.py` automatically creates a dedicated folder with clean, standardized naming:
```text
output/
└── Created_Scenes_pantry/
    ├── Created_Scenes_pantry.ply                # Colored metric 3D point cloud
    ├── Created_Scenes_pantry_splat.ply          # 17-column 3D Gaussian Splatting binary PLY (if --splat)
    ├── Created_Scenes_pantry_predictions.npz    # Raw predictions archive (depths, extrinsics, confidences)
    ├── Created_Scenes_pantry_camera_poses.json  # Formatted camera translations and 3x3 rotation matrices
    └── Created_Scenes_pantry.glb                # Trimesh 3D scene with camera frustums (if --export_glb)
```

### Comprehensive CLI Reference

#### 📁 Input & Model Configuration
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input`, `-i` | `str` | *Required* | Path to a single scene folder or parent folder containing multiple scene subfolders. |
| `--output`, `-o` | `str` | `./output` | Root output directory for generated scene folders and assets. |
| `--model_path`, `-m` | `str` | `None` | Path to local model weights (`.pt`). Auto-downloads from HuggingFace if omitted or not found. |
| `--device` | `str` | `None` | Compute device (`cuda`, `cpu`, or auto-detect if `None`). |

#### 🖼️ Argus Native Preprocessing & Resolution
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--img_size` | `int` | `560` | Input panoramic target width (height = `img_size // 2`, standard Argus resolution is `280x560`). |
| `--crop_ratio` | `float` | `0.15` | Vertical crop ratio on top and bottom of equirectangular panoramas (0.0 to 0.45) to exclude ceiling/tripod nadir artifacts. |
| `--dense` | `flag` | `False` | Enable high-resolution guided depth upsampling to match input photo dimensions with true RGB color fidelity. |
| `--dense-resize`, `--dense_resize` | `str` | `2048` | Maximum width resolution cap for dense upsampling (`2048`/`2k`, `4096`/`4k`, or `orig`/`0` for uncapped). |

#### 🧹 Point Cloud & Filtering Controls
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--conf_thres` | `float` | `5.0` | Confidence cutoff percentile (0.0–100.0) to filter noisy points. Set `0.0` to preserve all valid points. |
| `--ceiling_remove` | `float` | `0.0` | Percentile of top Y-coordinate ceiling points to prune (0.0–100.0). Set `25.0` for an open-top dollhouse floorplan view. |

#### 🌟 3D Gaussian Splatting (`--splat`)
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--splat` | `flag` | `False` | Generate 17-column 3D Gaussian Splatting `<scene_name>_splat.ply` directly on GPU. |
| `--ply_is_indoor` / `--no-ply_is_indoor` | `bool` | `True` | Scene depth cutoff preset. Indoor mode caps depth at 15m; `--no-ply_is_indoor` extends cutoff to 80m for outdoor environments. |
| `--ply_stride` | `int` | `1` | Pixel subsampling stride for splats (`1` = full resolution, `2` = half resolution for faster web rendering). |
| `--ply_scale` | `float` | `1.2` | Global splat radius scale multiplier (guarantees smooth tangential surface overlap without gaps). |
| `--ply_thickness` | `float` | `0.2` | Splat disc thickness ratio relative to tangent radius. |
| `--ply_min_depth` | `float` | `0.1` | Near plane depth threshold in meters. |
| `--ply_max_depth` | `float` | `None` | Custom far plane depth cutoff in meters (overrides indoor 15m / outdoor 80m preset). |

#### 🧊 3D Scene & GLB Export
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--export_glb` | `flag` | `False` | Export interactive 3D trimesh scene as `<scene_name>.glb`. |
| `--show_cam` / `--no-show_cam` | `bool` | `True` | Include 3D color-coded camera cone frustums at estimated camera positions inside the GLB scene. |
| `--debug` | `flag` | `False` | Export per-panorama point clouds and splats inside `<scene>/debug/<scene> <idx>/` in aligned world coordinates. |
| `--nested` | `flag` | `False` | Enable hierarchical multi-step reconstruction for nested scene trees (e.g. `/villa/1/1`), eliminating VRAM spikes and hallucinations. |

### 🌳 Hierarchical Multi-Step Reconstruction (`--nested`)
For complex, multi-room architectural spaces (e.g., a villa with connected bedrooms and bathrooms), passing all panoramas simultaneously into Argus can exhaust GPU VRAM and cause attention hallucinations across non-covisible spaces.

With `--nested`, organize rooms into minimal complete subsets where child folders are named after the parent bridge image index:
```text
/villa/
├── 0.jpg                     (Villa - Main Center)
├── 1.jpg                     (Bedroom 1 - Bridge to /villa/1)
├── 2.jpg                     (Bedroom 2 - Bridge to /villa/2)
│
├── 1/                        (Bedroom 1)
│   ├── 0.jpg                 (Bedroom 1 - Local Center)
│   └── 1.jpg                 (Bathroom 1 - Bridge to /villa/1/1)
│       └── 1/                (Bathroom 1)
│           └── 0.jpg         (Bathroom 1 - Local Center)
```
Argus reconstructs the deepest subfolders first, computes $\operatorname{Sim}(3)$ coordinate alignment and median depth scaling across bridge panoramas, and merges all geometries into the root coordinate system.

```bash
uv run python infer.py \
  --input "/path/to/villa" \
  --output ./output \
  --nested \
  --dense-resize 4k \
  --splat \
  --ceiling_remove 25 \
  --debug
```

## Interactive Demo

Launch the Gradio demo for interactive 3D reconstruction and metric measurement:

```bash
# Model will be auto-downloaded from HuggingFace if not found locally
# (requires: hf auth login)
python demo_gradio.py

# Or specify a local model path
python demo_gradio.py --model_path models/argus_realsee3d.pt
```

The demo supports:
- Uploading multiple panoramic images
- Real-time 3D reconstruction with GLB export
- Interactive metric distance measurement between points
- Adjustable confidence thresholds and visualization options

## Evaluation

Evaluate on the Realsee3D benchmark:

```bash
cd evaluation
python eval.py \
  --model_path ../models/argus_realsee3d.pt \
  --dataset_path /path/to/Realsee3D \
  --split both \
  --ref
```

Metrics include camera pose accuracy, depth error, point map quality, and covisibility estimation.

## Training

Training supports multi-GPU distributed training:

```bash
cd training
torchrun --nproc_per_node=8 launch.py --config full
```

See [`training/config/full.yaml`](training/config/full.yaml) for the full training configuration.

## License

This project is licensed under the [Apache License 2.0](LICENSE). The pretrained model weights trained on RealSee3D are released under a **non-commercial license**, consistent with the [RealSee3D dataset license](https://github.com/realsee-developer/RealSee3D).

## Citation

```bibtex
@misc{li2026argusmetricpanoramic3d,
      title={Argus: Metric Panoramic 3D Reconstruction for Indoor Scenes}, 
      author={Xi Li and Linyuan Li and Yan Wu and Tong Rao and Kai Zhang and Xinchen Hui and Cihui Pan},
      year={2026},
      eprint={2606.30047},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.30047}, 
}
```

## Acknowledgements

Argus builds upon [VGGT](https://github.com/facebookresearch/vggt).
