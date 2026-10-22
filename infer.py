#!/usr/bin/env python3
"""
Argus: Metric Panoramic 3D Reconstruction Pipeline
Standalone inference script supporting:
- Single scene folder or parent folder containing multiple scene subfolders
- Configurable output directory with automatic folder creation
- Strict scene-based naming conventions:
    <output_dir>/<scene_name>/<scene_name>.ply
    <output_dir>/<scene_name>/<scene_name>_predictions.npz
    <output_dir>/<scene_name>/<scene_name>_camera_poses.json
- Standard (native 560px) and Dense (high-res guided depth upsampling) modes
- Platform-independent (Windows, Linux, Kaggle, Colab)
"""

import os
import sys
import glob
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import cv2
import torch
import numpy as np

# Argus package imports
from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360
from argus.utils.geometry import unproject_depth_to_world_points

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Argus 3D Panoramic Reconstruction CLI Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to an input scene folder or a parent folder containing multiple scene subfolders."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Root output directory (e.g. ./output or /kaggle/working/output). Created if not existing."
    )
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default=None,
        help="Path to model weights (.pt). Auto-downloads from HuggingFace if omitted or not found."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=560,
        help="Inference width for the model (height = img_size // 2). Default: 560."
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=0.15,
        help="Vertical crop ratio on top and bottom (0.0 to 0.45). Default: 0.15."
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        default=False,
        help="Enable high-resolution guided depth upsampling to match original image resolutions."
    )
    parser.add_argument(
        "--dense_resize", "--dense-resize",
        type=str,
        default="2048",
        help="Max width cap for dense upsampling (default: '2048' / '2k'; use '4096' / '4k' or '0' / 'orig' for uncapped)."
    )
    parser.add_argument(
        "--splat",
        action="store_true",
        default=False,
        help="Save 3D Gaussian Splatting (<scene_name>_splat.ply) file."
    )
    parser.add_argument(
        "--ply_is_indoor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scene preset (indoor: max 15m cutoff; outdoor: max 80m cutoff)."
    )
    parser.add_argument(
        "--ply_stride",
        type=int,
        default=1,
        help="Pixel sampling stride for splats (1 = full res, 2 = half res)."
    )
    parser.add_argument(
        "--ply_scale",
        type=float,
        default=1.2,
        help="Global splat radius scale multiplier. Default: 1.2."
    )
    parser.add_argument(
        "--ply_thickness",
        type=float,
        default=0.2,
        help="Splat disc thickness ratio. Default: 0.2."
    )
    parser.add_argument(
        "--ply_min_depth",
        type=float,
        default=0.1,
        help="Minimum distance threshold in meters. Default: 0.1."
    )
    parser.add_argument(
        "--ply_max_depth",
        type=float,
        default=None,
        help="Maximum distance cutoff in meters (15.0m indoor, 80.0m outdoor by default)."
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=5.0,
        help="Confidence cutoff percentile (0-100). Default: 5.0. Set 0.0 to disable confidence filtering."
    )
    parser.add_argument(
        "--ceiling_remove",
        type=float,
        default=0.0,
        help="Percentile of top Y-coordinate ceiling points to remove (0-100). Default: 0.0 (disabled)."
    )
    parser.add_argument(
        "--export_glb",
        action="store_true",
        default=False,
        help="Also export a 3D trimesh scene as .glb with camera meshes."
    )
    parser.add_argument(
        "--show_cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include 3D camera cones in the exported GLB scene (default: True)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or None for auto-detection."
    )
    return parser.parse_args()


# -------------------------- Scene Discovery --------------------------
def find_images_in_dir(directory: Path) -> List[Path]:
    """Find all image files directly in directory or in its 'images' subfolder."""
    candidates = []
    # Direct check
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            candidates.append(f)
    
    # If none found directly, check if there is an 'images' subfolder
    if not candidates:
        sub_img_dir = directory / "images"
        if sub_img_dir.is_dir():
            for f in sub_img_dir.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    candidates.append(f)

    return sorted(candidates, key=lambda p: p.name)


def discover_scenes(input_path_str: str) -> List[Tuple[str, List[Path]]]:
    """
    Discover scenes from input path.
    - If input_path directly contains images: single scene named after folder.
    - If input_path contains subfolders with images: multiple scenes named after each subfolder.
    """
    input_path = Path(input_path_str).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Case 1: Direct images found in the input folder
    direct_images = find_images_in_dir(input_path)
    if direct_images:
        scene_name = input_path.name
        return [(scene_name, direct_images)]

    # Case 2: Scan subdirectories
    scenes = []
    for sub in sorted(input_path.iterdir()):
        if sub.is_dir():
            sub_images = find_images_in_dir(sub)
            if sub_images:
                scenes.append((sub.name, sub_images))

    if not scenes:
        raise ValueError(
            f"No panoramic images ({', '.join(SUPPORTED_IMAGE_EXTS)}) found in '{input_path}' "
            f"or in any of its immediate subdirectories."
        )

    return scenes


# -------------------------- Model Loading --------------------------
def resolve_model_path(model_path: str = None) -> str:
    """Resolve model path: if exists locally use it, else auto-download from HuggingFace."""
    if model_path is not None and os.path.isfile(model_path):
        return model_path

    print("Checking HuggingFace for pre-trained weights (RealseeTechnology/argus-realsee3d)...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="RealseeTechnology/argus-realsee3d",
            filename="argus_realsee3d.pt",
        )
        print(f"Loaded weights from HuggingFace cache: {downloaded}")
        return downloaded
    except Exception as e:
        error_msg = str(e)
        if "GatedRepoError" in type(e).__name__ or "401" in error_msg:
            raise RuntimeError(
                "Access to the model repo requires authentication:\n"
                "  1. Run: hf auth login\n"
                "  2. Accept model license at: https://huggingface.co/RealseeTechnology/argus-realsee3d\n"
                "Or manually download 'argus_realsee3d.pt' and pass --model_path <path>."
            ) from e
        raise


def load_model(model_path: str, device: str) -> Argus:
    """Initialize Argus model and load state dict."""
    print(f"Loading Argus model onto {device}...")
    model = Argus(reorder_by_learning_ref=True, restore_metric_scale=True)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    print("Model initialized successfully.")
    return model


# -------------------------- Preprocessing --------------------------
def preprocess_images(
    image_paths: List[Path],
    target_width: int = 560,
    crop_ratio: float = 0.15
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Load images, resize to (target_width, target_width // 2), apply vertical crop, and normalize.
    Returns:
        tensor: [S, 3, H_crop, W] in [0, 1] range
        raw_images: List of original cropped RGB images (for dense mode / color binding)
    """
    pano_w = target_width
    pano_h = target_width // 2
    top_crop = int(pano_h * crop_ratio)
    bottom_crop = int(pano_h * (1.0 - crop_ratio))

    processed_list = []
    raw_images_list = []

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")

        raw_images_list.append(img_bgr)

        # Resize to model resolution if needed
        h, w = img_bgr.shape[:2]
        if w != pano_w or h != pano_h:
            resized_bgr = cv2.resize(img_bgr, (pano_w, pano_h), interpolation=cv2.INTER_AREA)
        else:
            resized_bgr = img_bgr

        # Crop vertically and convert BGR -> RGB
        cropped_rgb = resized_bgr[top_crop:bottom_crop, :, ::-1]
        cropped_rgb = np.ascontiguousarray(cropped_rgb)
        processed_list.append(cropped_rgb)

    stacked = np.stack(processed_list)  # (S, H_crop, W, 3)
    tensor = torch.from_numpy(stacked).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2)  # (S, 3, H_crop, W)

    return tensor, raw_images_list


# -------------------------- Point Cloud IO --------------------------
def write_ply(
    filepath: str,
    points: np.ndarray,
    colors: np.ndarray = None,
    binary: bool = True
):
    """
    Write 3D point cloud with RGB colors to PLY format.
    Fast binary little-endian writer with ASCII fallback.
    """
    num_pts = len(points)
    if num_pts == 0:
        print(f"Warning: No points to write to {filepath}")
        return

    has_colors = colors is not None and len(colors) == num_pts
    if has_colors and colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if binary:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_pts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
        )
        if has_colors:
            header += (
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
            )
        header += "end_header\n"

        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            if has_colors:
                dtype_spec = [
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")
                ]
                vertex_data = np.empty(num_pts, dtype=dtype_spec)
                vertex_data["x"] = points[:, 0]
                vertex_data["y"] = points[:, 1]
                vertex_data["z"] = points[:, 2]
                vertex_data["red"] = colors[:, 0]
                vertex_data["green"] = colors[:, 1]
                vertex_data["blue"] = colors[:, 2]
                vertex_data.tofile(f)
            else:
                points.astype("<f4").tofile(f)
    else:
        # ASCII writer
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_pts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
        )
        if has_colors:
            header += (
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
            )
        header += "end_header\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            if has_colors:
                combined = np.hstack([points, colors])
                np.savetxt(f, combined, fmt="%.4f %.4f %.4f %d %d %d")
            else:
                np.savetxt(f, points, fmt="%.4f %.4f %.4f")


# -------------------------- High-Res Guided Upsampling --------------------------
def parse_dense_resize(val: Union[str, int]) -> int:
    """Parse dense resize parameter ('2048', '2k', '4096', '4k', '0'/'orig' for uncapped)."""
    s = str(val).strip().lower()
    if s in ("0", "none", "orig", "original", "uncapped"):
        return 0
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def build_dense_pointcloud(
    raw_images_bgr: List[np.ndarray],
    low_depth: np.ndarray,       # (S, H_crop_low, W_low, 1)
    low_conf: np.ndarray,        # (S, H_crop_low, W_low)
    extrinsics: np.ndarray,      # (S, 3, 4)
    crop_ratio: float = 0.15,
    max_dense_width: int = 2048,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Upscale predicted depth maps to high-res panorama dimensions (capped by max_dense_width),
    unproject at full photographic resolution, and sample true original colors.
    """
    num_frames = len(raw_images_bgr)
    dense_depths = []
    dense_confs = []
    dense_colors = []

    # Target width from first original image (capped by max_dense_width if > 0)
    orig_h, orig_w = raw_images_bgr[0].shape[:2]
    if max_dense_width > 0 and orig_w > max_dense_width:
        ref_w = max_dense_width
        ref_h = max_dense_width // 2
        print(f"Capping dense upsampling resolution to {ref_w}x{ref_h} (source image is {orig_w}x{orig_h})...")
    else:
        ref_w = orig_w
        ref_h = orig_h
        print(f"Dense upsampling resolution: {ref_w}x{ref_h}...")

    top_crop = int(ref_h * crop_ratio)
    bottom_crop = int(ref_h * (1.0 - crop_ratio))
    h_crop_orig = bottom_crop - top_crop

    for i in range(num_frames):
        img_bgr = raw_images_bgr[i]
        curr_h, curr_w = img_bgr.shape[:2]

        # Resize original image to target (ref_w, ref_h) if capped
        if curr_w != ref_w or curr_h != ref_h:
            img_bgr = cv2.resize(img_bgr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

        # Crop original RGB and ensure positive contiguous strides (BGR -> RGB)
        cropped_bgr = img_bgr[top_crop:bottom_crop, :]
        cropped_rgb = np.ascontiguousarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

        # Upscale depth with bicubic interpolation
        d_slice = low_depth[i].squeeze(-1)
        d_upscaled = cv2.resize(d_slice, (ref_w, h_crop_orig), interpolation=cv2.INTER_CUBIC)

        # Upscale confidence with bilinear interpolation
        c_slice = low_conf[i]
        c_upscaled = cv2.resize(c_slice, (ref_w, h_crop_orig), interpolation=cv2.INTER_LINEAR)

        dense_depths.append(d_upscaled)
        dense_confs.append(c_upscaled)
        dense_colors.append(cropped_rgb)

    # Tensor format for Argus spherical unprojection
    depth_tensor = torch.from_numpy(np.stack(dense_depths)).unsqueeze(0).unsqueeze(-1).to(device)  # [1, S, H, W, 1]
    extri_tensor = torch.from_numpy(extrinsics).unsqueeze(0).to(device)                             # [1, S, 3, 4]

    with torch.no_grad():
        world_pts = unproject_depth_to_world_points(depth_tensor, extri_tensor, size=ref_w)
        world_pts = world_pts.squeeze(0).cpu().numpy()  # [S, H_crop, W, 3]

    pts_flat = world_pts.reshape(-1, 3)
    colors_flat = np.stack(dense_colors).reshape(-1, 3)
    confs_flat = np.stack(dense_confs).reshape(-1)

    return pts_flat, colors_flat, confs_flat, dense_depths, dense_colors


# -------------------------- 3D Gaussian Splatting (GPU Accelerated) --------------------------
def save_gaussian_splat_ply(
    filepath: Union[str, Path],
    points: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: Optional[torch.Tensor] = None
):
    """
    Exports points as standard 3D Gaussian Splatting binary PLY format.
    Compatible with SuperSplat, PlayCanvas, Luma AI, and WebGL 3DGS Viewers.
    100% GPU-accelerated binary assembly.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    N = points.shape[0]
    if N == 0:
        print(f"Warning: No Gaussians to save for {filepath}")
        return

    device = points.device
    pts = points.to(dtype=torch.float32)

    # Colors to Spherical Harmonics DC on GPU
    colors = colors.to(device=device, dtype=torch.float32)
    sh_dc = ((colors / 255.0) - 0.5) / 0.28209479177387814

    # Normals (zeros)
    normals = torch.zeros((N, 3), dtype=torch.float32, device=device)

    # Opacities
    if opacities is None:
        opacities = torch.full((N, 1), 4.5, dtype=torch.float32, device=device)
    else:
        opacities = opacities.to(device=device, dtype=torch.float32)
        if opacities.ndim == 1:
            opacities = opacities.unsqueeze(-1)

    scales = scales.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)

    # Assemble 17 float32 columns in 1 GPU kernel:
    # [x,y,z, nx,ny,nz, fdc0,fdc1,fdc2, opacity, sc0,sc1,sc2, rot0,rot1,rot2,rot3]
    packed = torch.cat([pts, normals, sh_dc, opacities, scales, quats], dim=-1)
    raw_bytes = packed.contiguous().cpu().numpy().tobytes()

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(raw_bytes)


def build_multiview_gaussian_splats(
    depth_maps: List[torch.Tensor],    # List of [H, W] float32 depth tensors
    rgb_images: List[torch.Tensor],    # List of [H, W, 3] uint8/float32 RGB tensors
    extrinsics: torch.Tensor,          # [S, 3, 4] or [S, 4, 4]
    stride: int = 1,
    is_indoor: bool = True,
    global_scale: float = 1.2,
    disc_thickness: float = 0.2,
    min_depth: float = 0.1,
    max_depth: Optional[float] = None,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    100% GPU-Accelerated conversion of multi-view panoramic depth and RGB images into
    unified, world-space 3D Gaussian Splats.
    """
    if max_depth is None:
        max_depth = 15.0 if is_indoor else 80.0

    all_pts, all_rgb, all_scales, all_quats = [], [], [], []
    num_views = len(depth_maps)

    for i in range(num_views):
        d_t = depth_maps[i].to(device=device, dtype=torch.float32)
        rgb_t = rgb_images[i].to(device=device)

        if stride > 1:
            d_t = d_t[::stride, ::stride]
            rgb_t = rgb_t[::stride, ::stride]

        H, W = d_t.shape[:2]

        u = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) / W
        v = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) / H
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        theta = (1.0 - u_grid) * (2.0 * torch.pi)
        phi = v_grid * torch.pi

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        dx = sin_phi * cos_theta
        dy = sin_phi * sin_theta
        dz = cos_phi

        dirs = torch.stack([dx, dy, dz], dim=-1)  # [H, W, 3] local ray directions
        pts_cam = dirs * d_t.unsqueeze(-1)        # [H, W, 3] local 3D points

        t1 = torch.stack([-sin_theta, cos_theta, torch.zeros_like(theta)], dim=-1)
        t2 = torch.stack([cos_phi * cos_theta, cos_phi * sin_theta, -sin_phi], dim=-1)

        d_theta = (2.0 * torch.pi) / W
        d_phi = torch.pi / H

        s1 = d_t * (d_theta * torch.clamp(sin_phi, min=1e-3)) * global_scale
        s2 = d_t * (d_phi * global_scale)
        s3 = disc_thickness * torch.minimum(s1, s2)

        scales = torch.stack([s1, s2, s3], dim=-1)
        log_scales = torch.log(torch.clamp(scales, min=1e-5, max=1e2))

        # World transform via Extrinsics [R | t]
        R = extrinsics[i, :3, :3].to(device=device, dtype=torch.float32)  # [3, 3]
        t = extrinsics[i, :3, 3].to(device=device, dtype=torch.float32)   # [3]

        # P_world = P_cam @ R.T + t
        pts_world = torch.matmul(pts_cam, R.T) + t

        # Rotate tangent frame into world coordinates
        t1_w = torch.matmul(t1, R.T)
        t2_w = torch.matmul(t2, R.T)
        dirs_w = torch.matmul(dirs, R.T)

        R00, R01, R02 = t1_w[..., 0], t2_w[..., 0], dirs_w[..., 0]
        R10, R11, R12 = t1_w[..., 1], t2_w[..., 1], dirs_w[..., 1]
        R20, R21, R22 = t1_w[..., 2], t2_w[..., 2], dirs_w[..., 2]

        tr = R00 + R11 + R22
        qw = torch.sqrt(torch.clamp(1.0 + tr, min=0.0)) / 2.0
        denom = 4.0 * torch.clamp(qw, min=1e-6)
        qx = (R21 - R12) / denom
        qy = (R02 - R20) / denom
        qz = (R10 - R01) / denom

        quats = torch.stack([qw, qx, qy, qz], dim=-1)
        quats = quats / torch.clamp(torch.norm(quats, dim=-1, keepdim=True), min=1e-6)

        valid = torch.isfinite(d_t) & (d_t >= min_depth) & (d_t <= max_depth)

        all_pts.append(pts_world[valid])
        all_rgb.append(rgb_t[valid])
        all_scales.append(log_scales[valid])
        all_quats.append(quats[valid])

    flat_pts = torch.cat(all_pts, dim=0)
    flat_rgb = torch.cat(all_rgb, dim=0)
    flat_scales = torch.cat(all_scales, dim=0)
    flat_quats = torch.cat(all_quats, dim=0)

    return flat_pts, flat_rgb, flat_scales, flat_quats


# -------------------------- 3D Visualization Helpers --------------------------
def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape) -> np.ndarray:
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend([
            (v1, v2, v2_offset),
            (v1, v1_offset, v3),
            (v3_offset, v2, v3),
            (v1, v2, v2_offset_2),
            (v1, v1_offset_2, v3),
            (v3_offset_2, v2, v3),
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def add_camera_cones_to_scene(scene_3d, extrinsics_np: np.ndarray, scene_scale: float):
    """Add colored 3D camera cone meshes into a Trimesh scene."""
    try:
        import trimesh
        from scipy.spatial.transform import Rotation
        import matplotlib.cm as cm

        num_cameras = len(extrinsics_np)
        extrinsics_4x4 = np.zeros((num_cameras, 4, 4))
        extrinsics_4x4[:, :3, :4] = extrinsics_np
        extrinsics_4x4[:, 3, 3] = 1.0

        cam_width = scene_scale * 0.02
        cam_height = scene_scale * 0.02
        rot_45 = np.eye(4)
        rot_45[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        rot_45[2, 3] = -cam_height
        opengl_matrix = get_opengl_conversion_matrix()

        cone = trimesh.creation.cone(cam_width, cam_height, sections=4)
        slight_rot = np.eye(4)
        slight_rot[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

        vertices_combined = np.concatenate([
            cone.vertices,
            0.95 * cone.vertices,
            transform_points(slight_rot, cone.vertices)
        ])
        mesh_faces = compute_camera_faces(cone)

        try:
            cmap = cm.get_cmap("rainbow")
        except AttributeError:
            cmap = cm.rainbow

        for i in range(num_cameras):
            camera_to_world = extrinsics_4x4[i]
            rgba = cmap(i / max(1, num_cameras - 1))
            face_colors = tuple(int(255 * x) for x in rgba[:3])

            complete_t = camera_to_world @ opengl_matrix @ rot_45
            vertices_transformed = transform_points(complete_t, vertices_combined)
            cam_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
            cam_mesh.visual.face_colors[:, :3] = face_colors
            scene_3d.add_geometry(cam_mesh)
    except Exception as e:
        print(f"Notice: Could not generate camera cone meshes ({e}). Skipping camera cones.")


# -------------------------- Scene Processor --------------------------
def process_scene(
    scene_name: str,
    image_paths: List[Path],
    output_dir: Path,
    model: Argus,
    device: str,
    args: argparse.Namespace
):
    clean_scene_name = scene_name.strip().replace(" ", "_")
    print(f"\n========================================================")
    print(f"Processing Scene: {clean_scene_name} ({len(image_paths)} panoramas)")
    print(f"========================================================")

    scene_out_dir = output_dir / clean_scene_name
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess images
    print("Preprocessing panoramic images...")
    images_tensor, raw_images_bgr = preprocess_images(
        image_paths,
        target_width=args.img_size,
        crop_ratio=args.crop_ratio
    )
    images_tensor = images_tensor.to(device)

    # 2. Model Inference
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Running multi-view model inference (AMP dtype: {dtype})...")
    t0 = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
        predictions = model(images_tensor)

    if device == "cuda":
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - t0
    print(f"Model forward pass completed in {inf_time:.2f}s")

    # 3. Decode Camera Extrinsics
    print("Decoding camera extrinsics...")
    extrinsic_tensor, pose_conf = pose_encoding_to_extri360(pose_encoding=predictions["pose_enc"])
    predictions["extrinsic"] = extrinsic_tensor[:, :, :3, :]

    # Unproject standard depth to world coordinates
    predictions["world_points_from_depth"] = unproject_depth_to_world_points(
        predictions["depth"], predictions["extrinsic"], size=args.img_size
    )

    # Convert tensors to CPU numpy arrays (squeeze batch dimension)
    np_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            np_predictions[k] = v.cpu().float().numpy().squeeze(0)

    # 4. Save raw predictions NPZ (<clean_scene_name>_predictions.npz)
    npz_filename = f"{clean_scene_name}_predictions.npz"
    npz_path = scene_out_dir / npz_filename
    print(f"Saving predictions archive: {npz_path}")
    np.savez_compressed(npz_path, **np_predictions)

    # 5. Save Camera Poses JSON (<clean_scene_name>_camera_poses.json)
    poses_filename = f"{clean_scene_name}_camera_poses.json"
    poses_path = scene_out_dir / poses_filename
    extrinsics_np = np_predictions["extrinsic"]  # (S, 3, 4)
    camera_poses_data = {
        "scene_name": clean_scene_name,
        "num_cameras": len(image_paths),
        "reference_camera_index": int(np_predictions.get("ref_idx", 0)),
        "cameras": []
    }

    for idx, img_p in enumerate(image_paths):
        R = extrinsics_np[idx, :3, :3].tolist()
        t = extrinsics_np[idx, :3, 3].tolist()
        camera_poses_data["cameras"].append({
            "index": idx,
            "filename": img_p.name,
            "translation": t,
            "rotation_matrix": R
        })

    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(camera_poses_data, f, indent=2)
    print(f"Saved camera poses to: {poses_path}")

    # 6. Build Point Cloud (Standard or Dense)
    dense_depths = None
    dense_colors = None
    if args.dense:
        max_dense_w = parse_dense_resize(args.dense_resize)
        print(f"Building high-resolution DENSE point cloud via guided depth upsampling (cap: {args.dense_resize})...")
        vertices_3d, colors_rgb, conf, dense_depths, dense_colors = build_dense_pointcloud(
            raw_images_bgr=raw_images_bgr,
            low_depth=np_predictions["depth"],
            low_conf=np_predictions["depth_conf"],
            extrinsics=extrinsics_np,
            crop_ratio=args.crop_ratio,
            max_dense_width=max_dense_w,
            device=device
        )
    else:
        print("Building standard resolution point cloud...")
        vertices_3d = np_predictions["world_points_from_depth"].reshape(-1, 3)
        images_rgb = np_predictions["images"]
        if images_rgb.ndim == 4 and images_rgb.shape[1] == 3:  # NCHW -> NHWC
            images_rgb = np.transpose(images_rgb, (0, 2, 3, 1))
        colors_rgb = (images_rgb.reshape(-1, 3) * 255.0).astype(np.uint8)
        conf = np_predictions["depth_conf"].reshape(-1)

    raw_count = len(vertices_3d)

    # 7. Confidence & Ceiling Filtering
    if args.conf_thres > 0.0:
        conf_cut = np.percentile(conf, args.conf_thres)
        mask = (conf >= conf_cut) & (conf > 1e-5)
        vertices_3d = vertices_3d[mask]
        colors_rgb = colors_rgb[mask]

    if args.ceiling_remove > 0.0 and len(vertices_3d) > 0:
        y_cut = np.percentile(vertices_3d[:, 1], args.ceiling_remove)
        mask_ceil = vertices_3d[:, 1] > y_cut
        vertices_3d = vertices_3d[mask_ceil]
        colors_rgb = colors_rgb[mask_ceil]

    final_count = len(vertices_3d)
    print(f"Point count: {final_count:,} (pruned {raw_count - final_count:,} noisy/ceiling points)")

    # 8. Save PLY Point Cloud (<clean_scene_name>.ply)
    ply_filename = f"{clean_scene_name}.ply"
    ply_path = scene_out_dir / ply_filename
    print(f"Exporting colored point cloud to: {ply_path}")
    write_ply(str(ply_path), vertices_3d, colors_rgb, binary=True)

    # 9. Optional Gaussian Splatting Export (<clean_scene_name>_splat.ply)
    if args.splat:
        print("Generating 3D Gaussian Splats on GPU...")
        splat_filename = f"{clean_scene_name}_splat.ply"
        splat_path = scene_out_dir / splat_filename

        if args.dense and dense_depths is not None:
            splat_depths = [torch.from_numpy(np.ascontiguousarray(d)) for d in dense_depths]
            splat_rgbs = [torch.from_numpy(np.ascontiguousarray(c)) for c in dense_colors]
        else:
            splat_depths = [
                torch.from_numpy(np.ascontiguousarray(np_predictions["depth"][i].squeeze(-1)))
                for i in range(len(image_paths))
            ]
            splat_rgbs = [
                torch.from_numpy(np.ascontiguousarray((np_predictions["images"][i].transpose(1, 2, 0) * 255.0).astype(np.uint8)))
                for i in range(len(image_paths))
            ]

        extri_tensor = torch.from_numpy(np.ascontiguousarray(extrinsics_np))

        t_sp = time.perf_counter()
        sp_pts, sp_rgb, sp_sc, sp_qt = build_multiview_gaussian_splats(
            depth_maps=splat_depths,
            rgb_images=splat_rgbs,
            extrinsics=extri_tensor,
            stride=args.ply_stride,
            is_indoor=args.ply_is_indoor,
            global_scale=args.ply_scale,
            disc_thickness=args.ply_thickness,
            min_depth=args.ply_min_depth,
            max_depth=args.ply_max_depth,
            device=device
        )
        print(f"Generated {len(sp_pts):,} Gaussian Splats in {time.perf_counter() - t_sp:.2f}s")
        save_gaussian_splat_ply(
            filepath=splat_path,
            points=sp_pts,
            colors=sp_rgb,
            scales=sp_sc,
            quats=sp_qt
        )
        print(f"Exported Gaussian Splats to: {splat_path}")

    # 10. Optional GLB Export (<clean_scene_name>.glb)
    if args.export_glb:
        try:
            import trimesh
            glb_filename = f"{clean_scene_name}.glb"
            glb_path = scene_out_dir / glb_filename
            print(f"Exporting trimesh GLB scene: {glb_path}")
            scene_3d = trimesh.Scene()
            pcd = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
            scene_3d.add_geometry(pcd)

            if args.show_cam:
                scene_scale = float(np.max(np.abs(vertices_3d))) if len(vertices_3d) > 0 else 1.0
                add_camera_cones_to_scene(scene_3d, extrinsics_np, scene_scale)

            scene_3d.export(str(glb_path))
            print(f"Exported interactive GLB scene to: {glb_path}")
        except ImportError:
            print("Note: 'trimesh' not installed. Skipping .glb export.")

    print(f"Scene '{clean_scene_name}' completed successfully.")


# -------------------------- Main Routine --------------------------
def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Argus Pipeline Starting | Device: {device} | Output root: {args.output}")

    # 1. Discover scenes
    scenes = discover_scenes(args.input)
    print(f"Discovered {len(scenes)} scene(s) to process:")
    for s_name, s_imgs in scenes:
        print(f"  • {s_name}: {len(s_imgs)} panoramic images")

    # 2. Resolve weights & load model
    model_weight_path = resolve_model_path(args.model_path)
    model = load_model(model_weight_path, device=device)

    # 3. Process each scene
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for scene_name, image_paths in scenes:
        process_scene(
            scene_name=scene_name,
            image_paths=image_paths,
            output_dir=output_root,
            model=model,
            device=device,
            args=args
        )

    print(f"\n========================================================")
    print(f"All {len(scenes)} scene(s) processed in {time.time() - total_start:.2f}s.")
    print(f"Results stored under: {output_root}")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
