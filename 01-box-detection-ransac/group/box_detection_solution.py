#!/usr/bin/env python3
"""
Computer Vision Project, Summer 2026 - solution for Exercise 1.1 / 1.2 (Box Detection)

What this script does:
1. Loads all provided Kinect .mat examples
2. Visualizes amplitude image, distance image, and a subsampled point cloud
3. Implements plane fitting with a self-written RANSAC
4. Detects the floor plane
5. Cleans the floor mask with morphological operators
6. Detects the top plane of the box
7. Extracts the largest connected component as the actual box top
8. Estimates box height, length, and width
9. Saves result figures and a JSON summary
10. Saves an extra clean presentation-style visualization similar to the exercise sheet

Usage:
    python box_detection_solution.py
"""
from __future__ import annotations

import os
import json
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import ndimage


def fit_plane_svd(points: np.ndarray):
    """Fit a plane to 3D points using SVD.
    Plane form: n dot x = d, where ||n|| = 1.
    """
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = float(n @ centroid)
    if d < 0:
        n = -n
        d = -d
    return n.astype(np.float32), float(d)


def ransac_plane(
    points: np.ndarray,
    threshold: float,
    max_iterations: int,
    batch_size: int = 128,
    seed: int = 0,
    eval_points: int = 10000,
):
    """Estimate a dominant plane with self-written RANSAC."""
    rng = np.random.default_rng(seed)

    valid_mask = np.isfinite(points).all(axis=1) & (np.abs(points[:, 2]) > 1e-9)
    pts = points[valid_mask].astype(np.float32)
    if len(pts) < 3:
        raise ValueError("Not enough valid points for RANSAC.")

    if len(pts) > eval_points:
        eval_idx = rng.choice(len(pts), size=eval_points, replace=False)
        eval_pts = pts[eval_idx]
    else:
        eval_pts = pts

    best_count = -1
    best_n = None
    best_d = None

    n_batches = math.ceil(max_iterations / batch_size)
    for _ in range(n_batches):
        idx = rng.integers(0, len(eval_pts), size=(batch_size, 3))
        p1 = eval_pts[idx[:, 0]]
        p2 = eval_pts[idx[:, 1]]
        p3 = eval_pts[idx[:, 2]]

        normals = np.cross(p2 - p1, p3 - p1)
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 1e-6
        if not np.any(good):
            continue

        normals = normals[good] / norms[good, None]
        ds = np.sum(normals * p1[good], axis=1).astype(np.float32)

        distances = np.abs(eval_pts @ normals.T - ds[None, :])
        counts = np.sum(distances < threshold, axis=0)
        j = int(np.argmax(counts))

        if int(counts[j]) > best_count:
            best_count = int(counts[j])
            best_n = normals[j]
            best_d = float(ds[j])

            if best_count == len(eval_pts):
                break

    # Refit on full inlier set for a cleaner final plane
    full_distances = np.abs(pts @ best_n - best_d)
    inliers = full_distances < threshold
    ref_n, ref_d = fit_plane_svd(pts[inliers])

    full_distances = np.abs(pts @ ref_n - ref_d)
    inliers = full_distances < threshold

    return {
        "normal": ref_n,
        "d": float(ref_d),
        "valid_mask": valid_mask,
        "inlier_mask_valid": inliers,
        "count": int(inliers.sum()),
        "num_valid": int(len(pts)),
    }


def valid_inlier_mask_to_image(valid_flat, inlier_valid, shape_hw):
    """Map inliers from the valid-point list back to image coordinates."""
    H, W = shape_hw
    out = np.zeros(H * W, dtype=bool)
    valid_idx = np.flatnonzero(valid_flat)
    out[valid_idx[inlier_valid]] = True
    return out.reshape(H, W)


def largest_connected_component(mask: np.ndarray):
    labels, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, num + 1))
    best_label = int(np.argmax(sizes)) + 1
    return labels == best_label, int(sizes[best_label - 1])


def orthonormal_basis_from_normal(normal):
    normal = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v, normal


def oriented_box_dimensions(points, plane_normal):
    """Compute length/width on the top plane using a PCA-aligned 2D bounding box."""
    u, v, _ = orthonormal_basis_from_normal(plane_normal)
    centroid = points.mean(axis=0)
    centered = points - centroid

    uv = np.column_stack([centered @ u, centered @ v])
    cov = np.cov(uv.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    uv_rot = uv @ evecs
    mins = uv_rot.min(axis=0)
    maxs = uv_rot.max(axis=0)
    dims = maxs - mins

    corners2 = np.array([
        [mins[0], mins[1]],
        [maxs[0], mins[1]],
        [maxs[0], maxs[1]],
        [mins[0], maxs[1]],
    ], dtype=np.float32)

    basis2 = np.stack([u, v], axis=1) @ evecs
    corners3 = centroid + corners2 @ basis2.T

    length = float(dims[0])
    width = float(dims[1])
    if width > length:
        length, width = width, length

    return length, width, corners3


def load_example(example_idx: int, input_dir: str):
    data = sio.loadmat(os.path.join(input_dir, f"example{example_idx}kinect.mat"))
    return (
        data[f"amplitudes{example_idx}"],
        data[f"distances{example_idx}"],
        data[f"cloud{example_idx}"],
    )


def save_input_visualization(example_idx, A, D, PC, out_dir):
    fig = plt.figure(figsize=(15, 4.5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(A, cmap="gray")
    ax1.set_title(f"Example {example_idx}: amplitude image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    valid = PC[..., 2] != 0
    d_show = np.where(valid, D, np.nan)
    im = ax2.imshow(d_show, cmap="viridis")
    ax2.set_title(f"Example {example_idx}: distance image")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    pts = PC.reshape(-1, 3)
    valid_pts = pts[np.abs(pts[:, 2]) > 1e-9]
    step = max(1, len(valid_pts) // 10000)
    sub = valid_pts[::step]
    ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1)
    ax3.set_title(f"Example {example_idx}: subsampled point cloud")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    plt.tight_layout()
    path = os.path.join(out_dir, f"example{example_idx}_inputs.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_detection_visualization(example_idx, D, floor_mask, floor_mask_filtered, top_mask_raw, top_mask_box, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    valid = ~np.isnan(D)

    axes[0].imshow(np.where(valid, D, np.nan), cmap="viridis")
    axes[0].set_title("distance")
    axes[1].imshow(floor_mask, cmap="gray")
    axes[1].set_title("floor mask (raw)")
    axes[2].imshow(floor_mask_filtered, cmap="gray")
    axes[2].set_title("floor mask (filtered)")
    axes[3].imshow(top_mask_raw, cmap="gray")
    axes[3].set_title("top plane mask")
    vis = np.zeros((top_mask_box.shape[0], top_mask_box.shape[1], 3))

    # Blue background
    vis[:, :] = [0.0, 0.4, 1.0]

    # Green floor
    vis[floor_mask_filtered] = [0.6, 0.9, 0.4]

    # Red box top
    vis[top_mask_box] = [0.8, 0.0, 0.0]

    axes[4].imshow(vis)
    axes[4].set_title("final detection (colored)")
    axes[4].set_title("largest CC = box top")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"example{example_idx}_masks.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_3d_visualization(example_idx, PC, floor_mask_filtered, top_mask_box, corners, out_dir):
    """Original technical 3D scatter plot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    valid = PC[..., 2] != 0
    pts_all = PC[valid]
    step = max(1, len(pts_all) // 15000)
    pts_sub = pts_all[::step]
    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], s=1, alpha=0.15, label="scene")

    floor_pts = PC[floor_mask_filtered]
    step_floor = max(1, len(floor_pts) // 4000)
    floor_sub = floor_pts[::step_floor]
    ax.scatter(floor_sub[:, 0], floor_sub[:, 1], floor_sub[:, 2], s=3, label="floor")

    box_pts = PC[top_mask_box]
    step_box = max(1, len(box_pts) // 3000)
    box_sub = box_pts[::step_box]
    ax.scatter(box_sub[:, 0], box_sub[:, 1], box_sub[:, 2], s=6, label="box top")

    cyc = np.vstack([corners, corners[0]])
    ax.plot(cyc[:, 0], cyc[:, 1], cyc[:, 2], linewidth=2, label="corners")

    ax.set_title(f"Example {example_idx}: 3D result")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(out_dir, f"example{example_idx}_3d.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_presentation_visualization(example_idx, PC, floor_mask_filtered, top_mask_box, corners, out_dir):
    """
    Clean presentation-style 2D visualization closer to the exercise-sheet style:
    - floor in green
    - box top in red
    - top outline in cyan
    - optional labels
    """
    H, W, _ = PC.shape

    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:, :] = np.array([0.00, 0.40, 1.00], dtype=np.float32)  # blue
    floor_color = np.array([0.60, 0.93, 0.45], dtype=np.float32)   # green
    top_color = np.array([0.70, 0.00, 0.00], dtype=np.float32)     # red
    edge_color = np.array([0.00, 0.95, 1.00], dtype=np.float32)    # cyan

    img[floor_mask_filtered] = floor_color
    img[top_mask_box] = top_color

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img, origin="upper")

    # map 3D corners to nearest pixels on the top mask
    top_coords = np.argwhere(top_mask_box)
    top_pts = PC[top_mask_box]

    corner_pixels = []
    for c in corners:
        d = np.linalg.norm(top_pts - c[None, :], axis=1)
        idx = int(np.argmin(d))
        r, col = top_coords[idx]
        corner_pixels.append([col, r])

    corner_pixels = np.array(corner_pixels, dtype=float)
    cyc = np.vstack([corner_pixels, corner_pixels[0]])
    ax.plot(cyc[:, 0], cyc[:, 1], color=edge_color, linewidth=2.5)

    # simple labels similar to sample figure
    cx = corner_pixels[:, 0].mean()
    cy = corner_pixels[:, 1].mean()
    ax.text(cx, cy - 35, "top", color="black", fontsize=9, ha="center")
    ax.text(corner_pixels[:, 0].min() - 22, cy, "left", color="black", fontsize=9, ha="center")
    ax.text(corner_pixels[:, 0].max() + 22, cy, "right", color="black", fontsize=9, ha="center")
    ax.text(cx + 10, corner_pixels[:, 1].max() + 16, "bottom", color="black", fontsize=9, ha="center")

    ax.set_title(f"Example {example_idx}: floor, box and box corners")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"example{example_idx}_presentation.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def solve_example(example_idx: int, input_dir: str, out_dir: str):
    A, D, PC = load_example(example_idx, input_dir)
    H, W, _ = PC.shape

    save_input_visualization(example_idx, A, D, PC, out_dir)

    floor_threshold = 0.02 if example_idx == 1 else 0.03
    top_threshold = 0.015 if example_idx == 1 else 0.03

    floor = ransac_plane(
        PC.reshape(-1, 3),
        threshold=floor_threshold,
        max_iterations=300,
        batch_size=128,
        seed=example_idx,
        eval_points=6000,
    )

    floor_mask = valid_inlier_mask_to_image(
        floor["valid_mask"],
        floor["inlier_mask_valid"],
        (H, W),
    )

    # Morphological filtering of the floor mask:
    # closing fills small holes, opening removes small blobs/noise
    floor_mask_filtered = ndimage.binary_closing(floor_mask, structure=np.ones((7, 7)))
    floor_mask_filtered = ndimage.binary_opening(floor_mask_filtered, structure=np.ones((5, 5)))

    non_floor_mask = (~floor_mask_filtered) & (PC[..., 2] != 0)
    non_floor_points = PC[non_floor_mask].reshape(-1, 3)

    top = ransac_plane(
        non_floor_points,
        threshold=top_threshold,
        max_iterations=250,
        batch_size=96,
        seed=100 + example_idx,
        eval_points=min(5000, len(non_floor_points)),
    )

    top_local_mask = np.zeros(len(non_floor_points), dtype=bool)
    top_valid_idx = np.flatnonzero(top["valid_mask"])
    top_local_mask[top_valid_idx[top["inlier_mask_valid"]]] = True

    non_floor_coords = np.argwhere(non_floor_mask)
    top_mask_raw = np.zeros((H, W), dtype=bool)
    top_mask_raw[
        non_floor_coords[top_local_mask, 0],
        non_floor_coords[top_local_mask, 1]
    ] = True

    # Clean top plane mask, then keep only the largest connected component.
    top_mask_raw = ndimage.binary_opening(top_mask_raw, structure=np.ones((3, 3)))
    top_mask_raw = ndimage.binary_closing(top_mask_raw, structure=np.ones((7, 7)))
    top_mask_box, cc_size = largest_connected_component(top_mask_raw)

    box_points = PC[top_mask_box]
    top_n, top_d = fit_plane_svd(box_points)

    # Make sure both normals roughly point in the same direction
    if np.dot(top_n, floor["normal"]) < 0:
        top_n = -top_n
        top_d = -top_d

    # Height estimation:
    # 1) average distance from top-plane points to floor plane
    # 2) plane offset difference (good cross-check because planes are almost parallel)
    height_mean = float(np.mean(np.abs(box_points @ floor["normal"] - floor["d"])))
    height_planes = float(abs(top_d - floor["d"]))

    length, width, corners = oriented_box_dimensions(box_points, top_n)

    save_detection_visualization(
        example_idx,
        D,
        floor_mask,
        floor_mask_filtered,
        top_mask_raw,
        top_mask_box,
        out_dir,
    )

    save_3d_visualization(
        example_idx,
        PC,
        floor_mask_filtered,
        top_mask_box,
        corners,
        out_dir,
    )

    save_presentation_visualization(
        example_idx,
        PC,
        floor_mask_filtered,
        top_mask_box,
        corners,
        out_dir,
    )

    return {
        "example": example_idx,
        "floor_threshold_m": floor_threshold,
        "top_threshold_m": top_threshold,
        "floor_plane_normal": floor["normal"].tolist(),
        "floor_plane_d": float(floor["d"]),
        "top_plane_normal": top_n.tolist(),
        "top_plane_d": float(top_d),
        "num_floor_inliers": int(floor["count"]),
        "num_floor_valid_points": int(floor["num_valid"]),
        "top_component_pixels": int(cc_size),
        "height_m_mean_point_to_floor": height_mean,
        "height_m_plane_distance": height_planes,
        "length_m": float(length),
        "width_m": float(width),
        "corners_xyz_m": corners.tolist(),
    }


def main():
    input_dir = "data"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for example_idx in [1, 2, 3, 4]:
        result = solve_example(example_idx, input_dir, out_dir)
        summary.append(result)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Finished.")
    for item in summary:
        print(
            f"Example {item['example']}: "
            f"L={item['length_m']:.3f} m, "
            f"W={item['width_m']:.3f} m, "
            f"H={item['height_m_mean_point_to_floor']:.3f} m "
            f"(plane distance cross-check: {item['height_m_plane_distance']:.3f} m)"
        )


if __name__ == "__main__":
    main()
