import os
import re
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import cv2
import numpy as np
import torch

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def natural_sort_key(s: str):
    """Sort strings containing numbers in human order (e.g. '0', '1', '2', '10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


class NestedSceneNode:
    """Represents a node in the hierarchical nested panorama tree."""

    def __init__(self, folder_path: Path, rel_path: Path, name: str):
        self.folder_path = folder_path.resolve()
        self.rel_path = rel_path
        self.name = name
        self.images: List[Path] = []
        self.children: Dict[int, "NestedSceneNode"] = {}

    def __repr__(self) -> str:
        return f"<NestedSceneNode name='{self.name}' images={len(self.images)} children={list(self.children.keys())}>"


def parse_nested_scene_tree(root_dir: Path, rel_path: Optional[Path] = None) -> Optional[NestedSceneNode]:
    """
    Recursively discovers and constructs a NestedSceneNode tree starting from root_dir.
    
    Numbered subfolders (e.g. '1', '2') matching an image index are attached as child branches.
    For example:
      /villa/
        0.jpg
        1.jpg -> child branch /villa/1/
        2.jpg -> child branch /villa/2/
    """
    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        return None

    node_name = root_dir.name
    if rel_path is None:
        rel_path = Path(node_name)

    node = NestedSceneNode(folder_path=root_dir, rel_path=rel_path, name=node_name)

    # 1. Discover top-level images in this folder only
    candidate_images = []
    for item in root_dir.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            candidate_images.append(item)

    candidate_images.sort(key=lambda p: natural_sort_key(p.stem))
    node.images = candidate_images

    if not node.images:
        return None

    # Map stems and indices:
    stem_to_idx = {img.stem: idx for idx, img in enumerate(node.images)}
    norm_stem_to_idx = {re.sub(r'[\s_]+', '_', img.stem.lower()): idx for idx, img in enumerate(node.images)}

    # 2. Check for child subfolders corresponding to images
    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith((".", "debug", "output")):
            child_idx = None
            item_norm = re.sub(r'[\s_]+', '_', item.name.lower())

            # 1. Exact stem match takes highest precedence (e.g. folder '1' matches 1.jpg)
            if item.name in stem_to_idx:
                child_idx = stem_to_idx[item.name]
            # 2. Normalized stem match (e.g. folder 'bedroom 1' matches bedroom_1.jpg)
            elif item_norm in norm_stem_to_idx:
                child_idx = norm_stem_to_idx[item_norm]
            # 3. Direct digit index
            elif item.name.isdigit():
                val = int(item.name)
                if 0 <= val < len(node.images):
                    child_idx = val
                elif 1 <= val <= len(node.images):
                    child_idx = val - 1
            else:
                # 4. Trailing number matching
                nums = re.findall(r'\d+', item.name)
                if nums:
                    num_val = int(nums[-1])
                    if str(num_val) in stem_to_idx:
                        child_idx = stem_to_idx[str(num_val)]
                    elif 0 <= num_val < len(node.images):
                        child_idx = num_val

            if child_idx is not None and child_idx < len(node.images):
                child_rel = rel_path / item.name
                child_node = parse_nested_scene_tree(item, rel_path=child_rel)
                if child_node is not None and len(child_node.images) > 0:
                    node.children[child_idx] = child_node

    return node


# -------------------------- Sim(3) Coordinate Transformations --------------------------
def estimate_depth_scale(
    parent_depth: np.ndarray,
    child_depth: np.ndarray,
    min_scale: float = 0.5,
    max_scale: float = 2.0
) -> float:
    """
    Estimates the relative metric scale factor s between parent and child inferences
    using the median depth ratio across valid, overlapping pixels of the shared bridge panorama.
    """
    try:
        p_d = np.squeeze(parent_depth)
        c_d = np.squeeze(child_depth)

        # Match dimensions if needed (e.g. bicubic resize)
        if p_d.shape != c_d.shape:
            import cv2
            c_d = cv2.resize(c_d, (p_d.shape[1], p_d.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid = np.isfinite(p_d) & np.isfinite(c_d) & (p_d > 0.1) & (c_d > 0.1) & (p_d < 50.0) & (c_d < 50.0)
        if np.sum(valid) < 100:
            return 1.0

        ratios = p_d[valid] / c_d[valid]
        scale = float(np.median(ratios))
        return float(np.clip(scale, min_scale, max_scale))
    except Exception:
        return 1.0


def compute_sim3_transform(
    R_parent: np.ndarray,  # [3, 3] camera pose in parent
    t_parent: np.ndarray,  # [3]
    R_child: np.ndarray,   # [3, 3] camera pose in child
    t_child: np.ndarray,   # [3]
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the Sim(3) transformation (R, t, s) aligning child world coordinates into parent world coordinates:
      X_parent = s * R @ (X_child - t_child) + t_parent
               = s * (R @ X_child) + (t_parent - s * R @ t_child)
    where R = R_parent @ R_child.T.
    """
    R_align = R_parent @ R_child.T
    t_align = t_parent - scale * (R_align @ t_child)
    return R_align, t_align, scale


def transform_pointcloud_np(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """Transform points [N, 3] by Sim(3): s * (points @ R.T) + t."""
    if len(points) == 0:
        return points
    return scale * (points @ R.T) + t


def quat_to_matrix_torch(quats: torch.Tensor) -> torch.Tensor:
    """Convert [N, 4] quaternions [qw, qx, qy, qz] to [N, 3, 3] rotation matrices."""
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)

    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)

    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    return R


def transform_gaussian_splats_torch(
    points: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    R_align: np.ndarray,
    t_align: np.ndarray,
    scale: float,
    matrix_to_quaternion_fn: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform 3D Gaussian Splats by Sim(3) on GPU."""
    device = points.device
    R_t = torch.from_numpy(R_align).to(device=device, dtype=torch.float32)
    t_t = torch.from_numpy(t_align).to(device=device, dtype=torch.float32)

    # 1. Transform Centers
    pts_new = scale * torch.matmul(points, R_t.T) + t_t

    # 2. Transform Log-Scales: log(s_orig * scale) = log(s_orig) + log(scale)
    scales_new = scales + np.log(max(1e-4, scale))

    # 3. Transform Orientations: R_new = R_align @ R_orig
    R_orig = quat_to_matrix_torch(quats)  # [N, 3, 3]
    R_new = torch.matmul(R_t.unsqueeze(0), R_orig)  # [N, 3, 3]
    quats_new = matrix_to_quaternion_fn(R_new)

    return pts_new, scales_new, quats_new


def transform_camera_poses_list(
    cameras: List[dict],
    R_align: np.ndarray,
    t_align: np.ndarray,
    scale: float,
    prefix: str = ""
) -> List[dict]:
    """Transform camera pose dictionaries to parent coordinate frame."""
    transformed = []
    for c in cameras:
        R_orig = np.array(c["rotation_matrix"], dtype=np.float32)
        t_orig = np.array(c["translation"], dtype=np.float32)

        R_new = (R_align @ R_orig).tolist()
        t_new = (scale * (R_align @ t_orig) + t_align).tolist()

        c_copy = dict(c)
        c_copy["rotation_matrix"] = R_new
        c_copy["translation"] = t_new
        if prefix:
            c_copy["filename"] = f"{prefix}/{c.get('filename', '')}"
        transformed.append(c_copy)
    return transformed


# -------------------------- Post-Order Recursive Processor --------------------------
def process_nested_scene(
    node: NestedSceneNode,
    infer_fn: Callable[..., Dict[str, Any]],
    output_root: Path,
    args: Any,
    device: str,
    matrix_to_quaternion_fn: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, Any]:
    """
    Bottom-up hierarchical reconstruction:
      1. Post-order: recursively reconstructs deepest child subfolders first.
      2. Executes local Argus inference on the current node's small panoramic set.
      3. Aligns and merges each child node into the current node's world frame.
      4. Saves mirrored output files at each level, culminating in a fully merged model at the root.
    """
    clean_node_name = node.name.strip().replace(" ", "_")
    node_out_dir = output_root / node.rel_path
    node_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"Hierarchical Node: {node.rel_path} ('{node.name}') | {len(node.images)} panos")
    print(f"Child Branches: {list(node.children.keys())}")
    print(f"{'#'*65}")

    # 1. Post-order: Process all children first
    child_results: Dict[int, Dict[str, Any]] = {}
    for bridge_idx, child_node in sorted(node.children.items()):
        print(f"\n--> Descending into child branch [{bridge_idx}]: '{child_node.name}'...")
        child_res = process_nested_scene(
            node=child_node,
            infer_fn=infer_fn,
            output_root=output_root,
            args=args,
            device=device,
            matrix_to_quaternion_fn=matrix_to_quaternion_fn
        )
        child_results[bridge_idx] = child_res

    # 2. Infer current node's local panoramic image set
    print(f"\n--> Running inference on node '{node.name}' ({len(node.images)} images)...")
    node_res = infer_fn(
        scene_name=clean_node_name,
        image_paths=node.images,
        output_dir=node_out_dir,
        device=device,
        args=args
    )

    # If there are no children, this is a leaf node; return results directly
    if not child_results:
        print(f"<-- Leaf node '{node.name}' completed.")
        return node_res

    # 3. Merge children into current node coordinate frame
    print(f"\n--> Merging {len(child_results)} child branch(es) into '{node.name}' coordinate frame...")

    merged_points = []
    merged_colors = []
    merged_confs = []

    merged_splat_pts = []
    merged_splat_rgb = []
    merged_splat_sc = []
    merged_splat_qt = []

    # Identify which parent views have a child branch
    child_bridge_indices = set(child_results.keys())

    # Keep only parent cameras that are not bridge cameras (child will provide them with local detail)
    all_cameras = [
        dict(c) for c in node_res.get("camera_poses", {}).get("cameras", [])
        if c.get("index") not in child_bridge_indices
    ]

    # Collect points from parent node, excluding bridge views (child replaces them with higher detail)
    pts_per_view = node_res["pts_per_view"]
    colors_per_view = node_res["colors_per_view"]
    confs_per_view = node_res["confs_per_view"]
    conf_cut_val = node_res.get("conf_cut_val", None)
    y_cut = node_res.get("y_cut", None)

    for i in range(len(node.images)):
        if i in child_bridge_indices:
            # Skip: replaced by child's specialized local reconstruction
            continue

        p_v = pts_per_view[i].reshape(-1, 3)
        c_v = colors_per_view[i].reshape(-1, 3)
        conf_v = confs_per_view[i].reshape(-1)

        if conf_cut_val is not None:
            m = (conf_v >= conf_cut_val) & (conf_v > 1e-5)
            p_v = p_v[m]
            c_v = c_v[m]
            conf_v = conf_v[m]

        if args.ceiling_remove > 0.0 and y_cut is not None and len(p_v) > 0:
            m_ceil = p_v[:, 1] > y_cut
            p_v = p_v[m_ceil]
            c_v = c_v[m_ceil]
            conf_v = conf_v[m_ceil]

        merged_points.append(p_v)
        merged_colors.append(c_v)
        merged_confs.append(conf_v)

    # Collect parent splats if enabled
    if args.splat and "splat_per_view" in node_res and node_res["splat_per_view"] is not None:
        for i in range(len(node.images)):
            if i in child_bridge_indices:
                continue
            sp_p, sp_c, sp_s, sp_q = node_res["splat_per_view"][i]
            if args.ceiling_remove > 0.0 and y_cut is not None and len(sp_p) > 0:
                m_ceil = sp_p[:, 1] > y_cut
                sp_p = sp_p[m_ceil]
                sp_c = sp_c[m_ceil]
                sp_s = sp_s[m_ceil]
                sp_q = sp_q[m_ceil]

            merged_splat_pts.append(sp_p)
            merged_splat_rgb.append(sp_c)
            merged_splat_sc.append(sp_s)
            merged_splat_qt.append(sp_q)

    # Transform and accumulate each child branch
    extri_parent = node_res["extrinsics_np"]  # [S, 3, 4]
    parent_depths = node_res["dense_depths"]

    for bridge_idx, child_res in child_results.items():
        child_node = node.children[bridge_idx]
        extri_child = child_res["extrinsics_np"]

        # Identify bridge camera index in child:
        # Match by filename/stem with parent bridge image, or default to camera 0
        parent_bridge_img = node.images[bridge_idx]
        child_bridge_cam_idx = 0
        for c_idx, c_img in enumerate(child_node.images):
            if c_img.name == parent_bridge_img.name or c_img.stem == parent_bridge_img.stem:
                child_bridge_cam_idx = c_idx
                break

        # Pose of bridge image in parent vs child
        R_parent = extri_parent[bridge_idx, :3, :3]
        t_parent = extri_parent[bridge_idx, :3, 3]

        R_child = extri_child[child_bridge_cam_idx, :3, :3]
        t_child = extri_child[child_bridge_cam_idx, :3, 3]

        # Estimate scale ratio via shared bridge panorama depth
        p_depth = parent_depths[bridge_idx]
        c_depth = child_res["dense_depths"][child_bridge_cam_idx]
        scale_c2p = estimate_depth_scale(p_depth, c_depth)

        R_align, t_align, s_align = compute_sim3_transform(
            R_parent, t_parent, R_child, t_child, scale=scale_c2p
        )

        print(f"  • Bridge [{bridge_idx}] -> '{child_node.name}': scale={s_align:.3f}, translation dist={np.linalg.norm(t_align):.2f}m")

        # 1. Transform child merged point cloud
        child_pts = child_res["merged_vertices_3d"]
        child_clr = child_res["merged_colors_rgb"]
        child_cnf = child_res["merged_conf"]

        child_pts_transformed = transform_pointcloud_np(child_pts, R_align, t_align, scale=s_align)
        if args.ceiling_remove > 0.0 and y_cut is not None and len(child_pts_transformed) > 0:
            m_ceil = child_pts_transformed[:, 1] > y_cut
            child_pts_transformed = child_pts_transformed[m_ceil]
            child_clr = child_clr[m_ceil]
            child_cnf = child_cnf[m_ceil]

        merged_points.append(child_pts_transformed)
        merged_colors.append(child_clr)
        merged_confs.append(child_cnf)

        # 2. Transform child Gaussian splats if enabled
        if args.splat and child_res.get("merged_splat_points") is not None:
            c_sp_p = child_res["merged_splat_points"]
            c_sp_c = child_res["merged_splat_colors"]
            c_sp_s = child_res["merged_splat_scales"]
            c_sp_q = child_res["merged_splat_quats"]

            c_sp_p_t, c_sp_s_t, c_sp_q_t = transform_gaussian_splats_torch(
                points=c_sp_p,
                scales=c_sp_s,
                quats=c_sp_q,
                R_align=R_align,
                t_align=t_align,
                scale=s_align,
                matrix_to_quaternion_fn=matrix_to_quaternion_fn
            )

            if args.ceiling_remove > 0.0 and y_cut is not None and len(c_sp_p_t) > 0:
                m_ceil = c_sp_p_t[:, 1] > y_cut
                c_sp_p_t = c_sp_p_t[m_ceil]
                c_sp_c = c_sp_c[m_ceil]
                c_sp_s_t = c_sp_s_t[m_ceil]
                c_sp_q_t = c_sp_q_t[m_ceil]

            merged_splat_pts.append(c_sp_p_t)
            merged_splat_rgb.append(c_sp_c)
            merged_splat_sc.append(c_sp_s_t)
            merged_splat_qt.append(c_sp_q_t)

        # 3. Transform child camera poses and append
        child_cams = child_res.get("camera_poses", {}).get("cameras", [])
        cams_transformed = transform_camera_poses_list(
            child_cams, R_align, t_align, scale=s_align, prefix=str(child_node.rel_path)
        )
        all_cameras.extend(cams_transformed)

    # Re-index all cameras uniquely
    for new_idx, cam in enumerate(all_cameras):
        cam["index"] = new_idx

    # 4. Finalize merged structures for current node
    final_vertices = np.concatenate(merged_points, axis=0) if merged_points else np.empty((0, 3), dtype=np.float32)
    final_colors = np.concatenate(merged_colors, axis=0) if merged_colors else np.empty((0, 3), dtype=np.uint8)
    final_conf = np.concatenate(merged_confs, axis=0) if merged_confs else np.empty((0,), dtype=np.float32)

    # Strictly enforce ceiling cut on merged point cloud if ceiling_remove is requested
    if args.ceiling_remove > 0.0 and len(final_vertices) > 0:
        if y_cut is None:
            y_cut = float(np.percentile(final_vertices[:, 1], args.ceiling_remove))
        m_ceil = final_vertices[:, 1] > y_cut
        final_vertices = final_vertices[m_ceil]
        final_colors = final_colors[m_ceil]
        final_conf = final_conf[m_ceil]

    # Save merged Point Cloud
    ply_filename = f"{clean_node_name}.ply"
    ply_path = node_out_dir / ply_filename
    print(f"Exporting merged point cloud ({len(final_vertices):,} pts) to: {ply_path.relative_to(output_root)}")
    from infer import write_ply
    write_ply(str(ply_path), final_vertices, final_colors, binary=True)

    # Save merged Gaussian Splats if enabled
    final_splat_pts = None
    final_splat_rgb = None
    final_splat_sc = None
    final_splat_qt = None
    if args.splat and merged_splat_pts:
        final_splat_pts = torch.cat(merged_splat_pts, dim=0)
        final_splat_rgb = torch.cat(merged_splat_rgb, dim=0)
        final_splat_sc = torch.cat(merged_splat_sc, dim=0)
        final_splat_qt = torch.cat(merged_splat_qt, dim=0)

        # Strictly enforce ceiling cut on merged Gaussian Splats as well
        if args.ceiling_remove > 0.0 and y_cut is not None and len(final_splat_pts) > 0:
            m_ceil_sp = final_splat_pts[:, 1] > y_cut
            final_splat_pts = final_splat_pts[m_ceil_sp]
            final_splat_rgb = final_splat_rgb[m_ceil_sp]
            final_splat_sc = final_splat_sc[m_ceil_sp]
            final_splat_qt = final_splat_qt[m_ceil_sp]

        splat_filename = f"{clean_node_name}_splat.ply"
        splat_path = node_out_dir / splat_filename
        print(f"Exporting merged Gaussian Splats ({len(final_splat_pts):,} splats) to: {splat_path.relative_to(output_root)}")
        from infer import save_gaussian_splat_ply
        save_gaussian_splat_ply(
            filepath=splat_path,
            points=final_splat_pts,
            colors=final_splat_rgb,
            scales=final_splat_sc,
            quats=final_splat_qt
        )

    # Save updated camera poses JSON
    poses_filename = f"{clean_node_name}_camera_poses.json"
    poses_path = node_out_dir / poses_filename
    camera_poses_data = {
        "scene_name": clean_node_name,
        "num_cameras": len(all_cameras),
        "reference_camera_index": 0,
        "cameras": all_cameras
    }
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(camera_poses_data, f, indent=2)

    # Save merged predictions archive
    npz_filename = f"{clean_node_name}_predictions.npz"
    npz_path = node_out_dir / npz_filename
    np.savez_compressed(
        npz_path,
        world_points_from_depth=final_vertices,
        images_colors=final_colors,
        depth_conf=final_conf
    )

    # Optional GLB Export with camera cones
    if args.export_glb:
        try:
            import trimesh
            from infer import add_camera_cones_to_scene
            glb_filename = f"{clean_node_name}.glb"
            glb_path = node_out_dir / glb_filename
            print(f"Exporting trimesh GLB scene: {glb_path.relative_to(output_root)}")
            scene_3d = trimesh.Scene()
            pcd = trimesh.PointCloud(vertices=final_vertices, colors=final_colors)
            scene_3d.add_geometry(pcd)

            if args.show_cam and len(all_cameras) > 0:
                scene_scale = float(np.max(np.abs(final_vertices))) if len(final_vertices) > 0 else 1.0
                all_extrinsics = np.zeros((len(all_cameras), 3, 4), dtype=np.float32)
                for c_i, cam in enumerate(all_cameras):
                    all_extrinsics[c_i, :3, :3] = np.array(cam["rotation_matrix"], dtype=np.float32)
                    all_extrinsics[c_i, :3, 3] = np.array(cam["translation"], dtype=np.float32)
                add_camera_cones_to_scene(scene_3d, all_extrinsics, scene_scale)

            scene_3d.export(str(glb_path))
            print(f"Exported interactive GLB scene to: {glb_path.relative_to(output_root)}")
        except ImportError:
            print("Note: 'trimesh' not installed. Skipping .glb export.")

    # Package output dictionary for return
    node_res["merged_vertices_3d"] = final_vertices
    node_res["merged_colors_rgb"] = final_colors
    node_res["merged_conf"] = final_conf
    node_res["camera_poses"] = camera_poses_data
    node_res["merged_splat_points"] = final_splat_pts
    node_res["merged_splat_colors"] = final_splat_rgb
    node_res["merged_splat_scales"] = final_splat_sc
    node_res["merged_splat_quats"] = final_splat_qt

    return node_res
