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
# Basic reconstruction (point cloud + camera poses + predictions)
python infer.py --input path/to/scene_folder --output ./output

# High-resolution dense reconstruction (capped to 2K by default)
python infer.py --input path/to/scene_folder --output ./output --dense

# Generate photorealistic 3D Gaussian Splats (.ply) for SuperSplat / PlayCanvas / Luma
python infer.py --input path/to/scene_folder --output ./output --splat --ply_scale 1.2

# Batch process multiple scenes under a parent directory on Linux / Kaggle / Colab
python infer.py --input /path/to/scenes_parent_folder --output /path/to/output --dense --splat
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

### Key CLI Flags
| Flag | Description | Default |
| :--- | :--- | :---: |
| `--input`, `-i` | Path to a single scene folder or parent folder containing multiple scene subfolders | *Required* |
| `--output`, `-o` | Root output directory (e.g., `./output` or `/kaggle/working/output`) | `./output` |
| `--dense` | Enable guided depth upsampling to match high-res photo dimensions with true RGB colors | `False` |
| `--dense-resize` | Max width resolution cap for dense upsampling (`2048`/`2k`, `4096`/`4k`, or `orig`/`0`) | `2048` |
| `--splat` | Generate 3D Gaussian Splatting `<scene_name>_splat.ply` on GPU | `False` |
| `--ply_scale` | Global splat radius scale multiplier (ensures smooth surface overlap) | `1.2` |
| `--ply_thickness` | Splat disc thickness ratio relative to tangent radius | `0.2` |
| `--conf_thres` | Confidence cutoff percentile (set `0.0` to preserve all valid points) | `5.0` |
| `--ceiling_remove` | Percentile of top Y points to remove as ceiling (set `0.0` to keep ceiling) | `0.0` |
| `--export_glb` | Also export interactive `.glb` with 3D camera frustums | `False` |

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
