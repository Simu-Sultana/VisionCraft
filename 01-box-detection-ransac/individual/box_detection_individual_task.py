#!/usr/bin/env python3
"""
Computer Vision Project - Individual Exercise 1: Box Detection

This script reuses the group box-detection pipeline and extends it for the
individual task. It implements and evaluates three self-written plane-fitting
variants:

1. Standard RANSAC
2. MLESAC-style scoring
3. Preemptive RANSAC

It also includes a compact hyperparameter sweep, method-comparison plots,
runtime analysis, and a simple hypothesis test for the speed comparison.

Expected project layout:
    ProjectCV_EX_1/
    ├── box_detection_individual.py
    ├── data/
    │   ├── example1kinect.mat
    │   ├── example2kinect.mat
    │   ├── example3kinect.mat
    │   └── example4kinect.mat
    └── results_individual/

Run the full individual evaluation:
    python3 box_detection_individual.py --mode full

Run only the method comparison:
    python3 box_detection_individual.py --mode compare

The code intentionally does not use scikit-learn's RANSAC implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is normally available in this exercise
    stats = None


# -----------------------------------------------------------------------------
# Configuration and result containers
# -----------------------------------------------------------------------------

@dataclass
class RansacConfig:
    """Parameters used by all RANSAC variants."""

    threshold: float = 0.025
    max_iterations: int = 300
    batch_size: int = 128
    seed: int = 0
    eval_points: int = 6000
    confidence: float = 0.999
    adaptive: bool = True
    refine_iterations: int = 5

    # MLESAC-specific parameters
    mlesac_sigma_factor: float = 2.5
    mlesac_inlier_prior: float = 0.5

    # Preemptive-RANSAC-specific parameters
    preemptive_candidates: int = 512
    preemptive_block_size: int = 256
    preemptive_min_candidates: int = 4


@dataclass
class PlaneResult:
    normal: List[float]
    d: float
    valid_mask: np.ndarray
    inlier_mask_valid: np.ndarray
    count: int
    num_valid: int
    method: str
    actual_iterations: int
    score: float
    runtime_s: float


@dataclass
class DetectionResult:
    example: int
    method: str
    seed: int
    floor_threshold_m: float
    top_threshold_m: float
    max_iterations: int
    floor_plane_normal: List[float]
    floor_plane_d: float
    top_plane_normal: List[float]
    top_plane_d: float
    num_floor_inliers: int
    num_floor_valid_points: int
    num_top_inliers: int
    num_top_valid_points: int
    top_component_pixels: int
    normal_parallelism_abs_dot: float
    height_m_mean_point_to_floor: float
    height_m_median_point_to_floor: float
    height_m_plane_distance: float
    length_m: float
    width_m: float
    quality_score: float
    floor_runtime_s: float
    top_runtime_s: float
    total_runtime_s: float
    floor_iterations: int
    top_iterations: int


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a plane to 3D points using SVD.

    Plane representation: n dot x = d, with ||n|| = 1.
    """
    if points.shape[0] < 3:
        raise ValueError("At least three points are required to fit a plane.")

    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(n @ centroid)

    # Keep a deterministic sign where possible. The sign may later be aligned
    # to the floor normal when comparing two planes.
    if d < 0:
        n = -n
        d = -d
    return n.astype(np.float32), float(d)


def prepare_valid_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove invalid Kinect points.

    The exercise states that invalid point-cloud measurements can be identified
    by z == 0. We also remove non-finite values.
    """
    flat = points.reshape(-1, 3)
    valid_mask = np.isfinite(flat).all(axis=1) & (np.abs(flat[:, 2]) > 1e-9)
    return valid_mask, flat[valid_mask].astype(np.float32)


def sample_eval_points(pts: np.ndarray, rng: np.random.Generator, eval_points: int) -> np.ndarray:
    """Subsample points for candidate scoring."""
    if len(pts) <= eval_points:
        return pts
    idx = rng.choice(len(pts), size=eval_points, replace=False)
    return pts[idx]


def generate_plane_candidates(
    pts: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
    max_attempt_multiplier: int = 12,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Generate non-degenerate plane candidates from random point triples."""
    normals: List[np.ndarray] = []
    ds: List[np.ndarray] = []
    attempts = 0
    max_attempts = max(n_candidates * max_attempt_multiplier, n_candidates + 32)

    while sum(len(x) for x in ds) < n_candidates and attempts < max_attempts:
        need = n_candidates - sum(len(x) for x in ds)
        batch = max(need * 2, 64)
        idx = rng.integers(0, len(pts), size=(batch, 3))
        p1 = pts[idx[:, 0]]
        p2 = pts[idx[:, 1]]
        p3 = pts[idx[:, 2]]

        cand_normals = np.cross(p2 - p1, p3 - p1)
        norms = np.linalg.norm(cand_normals, axis=1)
        good = norms > 1e-6
        if np.any(good):
            cand_normals = cand_normals[good] / norms[good, None]
            cand_ds = np.sum(cand_normals * p1[good], axis=1).astype(np.float32)
            normals.append(cand_normals.astype(np.float32))
            ds.append(cand_ds.astype(np.float32))
        attempts += batch

    if not normals:
        raise ValueError("Could not generate a valid plane candidate.")

    normals_all = np.vstack(normals)[:n_candidates]
    ds_all = np.concatenate(ds)[:n_candidates]
    return normals_all, ds_all, attempts


def consensus_scores(eval_pts: np.ndarray, normals: np.ndarray, ds: np.ndarray, threshold: float) -> np.ndarray:
    """Count inliers for many candidate planes."""
    distances = np.abs(eval_pts @ normals.T - ds[None, :])
    return np.sum(distances < threshold, axis=0).astype(np.float64)


def mlesac_scores(
    eval_pts: np.ndarray,
    normals: np.ndarray,
    ds: np.ndarray,
    threshold: float,
    sigma_factor: float = 2.5,
    inlier_prior: float = 0.5,
) -> np.ndarray:
    """Compute an MLESAC-style log-likelihood score for each candidate.

    Standard RANSAC only counts points inside the threshold. MLESAC scores the
    complete residual distribution. Small residuals are rewarded more strongly
    than residuals just below the threshold, which usually makes the selected
    model less sensitive to the exact inlier threshold.
    """
    residuals = np.abs(eval_pts @ normals.T - ds[None, :])
    sigma = max(threshold / sigma_factor, 1e-6)

    # The outlier residual density is modelled as uniform over the observed
    # residual range. This is a practical approximation for this exercise.
    residual_range = max(float(np.percentile(residuals, 95)), threshold * 4.0, 1e-3)
    outlier_density = 1.0 / residual_range

    gaussian_density = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(
        -0.5 * (residuals / sigma) ** 2
    )
    mixture = inlier_prior * gaussian_density + (1.0 - inlier_prior) * outlier_density
    return np.sum(np.log(mixture + 1e-12), axis=0).astype(np.float64)


def refine_plane_iteratively(
    pts: np.ndarray,
    normal: np.ndarray,
    d: float,
    threshold: float,
    max_refine_iterations: int = 5,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Refit a plane by repeatedly recomputing inliers and applying SVD."""
    inliers = np.abs(pts @ normal - d) < threshold
    if inliers.sum() < 3:
        return normal.astype(np.float32), float(d), inliers

    previous: Optional[np.ndarray] = None
    for _ in range(max_refine_iterations):
        normal, d = fit_plane_svd(pts[inliers])
        new_inliers = np.abs(pts @ normal - d) < threshold
        if previous is not None and np.array_equal(new_inliers, previous):
            inliers = new_inliers
            break
        previous = inliers
        inliers = new_inliers
        if inliers.sum() < 3:
            break

    return normal.astype(np.float32), float(d), inliers


def adaptive_required_iterations(inlier_ratio: float, sample_size: int = 3, confidence: float = 0.999) -> float:
    """Compute the RANSAC iteration estimate for a desired confidence."""
    eps = float(np.clip(inlier_ratio, 1e-6, 1.0 - 1e-9))
    denominator = math.log(1.0 - eps ** sample_size)
    if abs(denominator) < 1e-12:
        return 1.0
    return math.log(1.0 - confidence) / denominator


# -----------------------------------------------------------------------------
# Three RANSAC variants
# -----------------------------------------------------------------------------

def ransac_plane_standard(points: np.ndarray, config: RansacConfig, method_name: str = "ransac") -> PlaneResult:
    """Standard self-written RANSAC with optional adaptive stopping."""
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    valid_mask, pts = prepare_valid_points(points)
    if len(pts) < 3:
        raise ValueError("Not enough valid points for RANSAC.")

    eval_pts = sample_eval_points(pts, rng, config.eval_points)
    best_count = -1
    best_score = -np.inf
    best_n: Optional[np.ndarray] = None
    best_d: Optional[float] = None
    required_iterations = float(config.max_iterations)
    actual_iterations = 0

    while actual_iterations < config.max_iterations:
        current_batch = min(config.batch_size, config.max_iterations - actual_iterations)
        normals, ds, _ = generate_plane_candidates(eval_pts, current_batch, rng)
        scores = consensus_scores(eval_pts, normals, ds, config.threshold)
        j = int(np.argmax(scores))

        if scores[j] > best_count:
            best_count = int(scores[j])
            best_score = float(scores[j])
            best_n = normals[j]
            best_d = float(ds[j])

            if config.adaptive:
                ratio = best_count / max(len(eval_pts), 1)
                required_iterations = min(
                    required_iterations,
                    adaptive_required_iterations(ratio, sample_size=3, confidence=config.confidence),
                )

        actual_iterations += current_batch

        if best_count == len(eval_pts):
            break
        if config.adaptive and actual_iterations >= required_iterations:
            break

    if best_n is None or best_d is None:
        raise RuntimeError("RANSAC failed to find a plane model.")

    ref_n, ref_d, inliers = refine_plane_iteratively(
        pts, best_n, best_d, config.threshold, config.refine_iterations
    )
    runtime = time.perf_counter() - start
    return PlaneResult(
        normal=ref_n.tolist(),
        d=float(ref_d),
        valid_mask=valid_mask,
        inlier_mask_valid=inliers,
        count=int(inliers.sum()),
        num_valid=int(len(pts)),
        method=method_name,
        actual_iterations=int(actual_iterations),
        score=float(best_score),
        runtime_s=float(runtime),
    )


def ransac_plane_mlesac(points: np.ndarray, config: RansacConfig) -> PlaneResult:
    """MLESAC-style plane fitting.

    Candidate generation is the same as RANSAC, but the selected model is the
    one with the best residual likelihood, not necessarily the largest binary
    inlier count.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    valid_mask, pts = prepare_valid_points(points)
    if len(pts) < 3:
        raise ValueError("Not enough valid points for MLESAC.")

    eval_pts = sample_eval_points(pts, rng, config.eval_points)
    best_score = -np.inf
    best_n: Optional[np.ndarray] = None
    best_d: Optional[float] = None
    actual_iterations = 0

    while actual_iterations < config.max_iterations:
        current_batch = min(config.batch_size, config.max_iterations - actual_iterations)
        normals, ds, _ = generate_plane_candidates(eval_pts, current_batch, rng)
        scores = mlesac_scores(
            eval_pts,
            normals,
            ds,
            threshold=config.threshold,
            sigma_factor=config.mlesac_sigma_factor,
            inlier_prior=config.mlesac_inlier_prior,
        )
        j = int(np.argmax(scores))
        if scores[j] > best_score:
            best_score = float(scores[j])
            best_n = normals[j]
            best_d = float(ds[j])
        actual_iterations += current_batch

    if best_n is None or best_d is None:
        raise RuntimeError("MLESAC failed to find a plane model.")

    ref_n, ref_d, inliers = refine_plane_iteratively(
        pts, best_n, best_d, config.threshold, config.refine_iterations
    )
    runtime = time.perf_counter() - start
    return PlaneResult(
        normal=ref_n.tolist(),
        d=float(ref_d),
        valid_mask=valid_mask,
        inlier_mask_valid=inliers,
        count=int(inliers.sum()),
        num_valid=int(len(pts)),
        method="mlesac",
        actual_iterations=int(actual_iterations),
        score=float(best_score),
        runtime_s=float(runtime),
    )


def ransac_plane_preemptive(points: np.ndarray, config: RansacConfig) -> PlaneResult:
    """Preemptive RANSAC plane fitting.

    A fixed pool of candidate planes is generated first. Instead of scoring all
    candidates on all points immediately, candidates are evaluated block by
    block. After every block, the worst half is removed. This avoids spending a
    full scoring pass on obviously weak candidates.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    valid_mask, pts = prepare_valid_points(points)
    if len(pts) < 3:
        raise ValueError("Not enough valid points for preemptive RANSAC.")

    eval_pts = sample_eval_points(pts, rng, config.eval_points)
    rng.shuffle(eval_pts)

    normals, ds, attempts = generate_plane_candidates(
        eval_pts,
        config.preemptive_candidates,
        rng,
    )
    alive = np.arange(len(normals))
    accumulated_scores = np.zeros(len(normals), dtype=np.float64)
    evaluated_points = 0

    for start_idx in range(0, len(eval_pts), config.preemptive_block_size):
        block = eval_pts[start_idx:start_idx + config.preemptive_block_size]
        if len(block) == 0 or len(alive) <= config.preemptive_min_candidates:
            break

        block_scores = consensus_scores(block, normals[alive], ds[alive], config.threshold)
        accumulated_scores[alive] += block_scores
        evaluated_points += len(block)

        # Prune only after scoring the current block. This is the key point that
        # avoids the common mistake of removing candidates before they have been
        # evaluated on comparable evidence.
        keep_count = max(config.preemptive_min_candidates, int(math.ceil(len(alive) / 2.0)))
        order = np.argsort(accumulated_scores[alive])[::-1]
        alive = alive[order[:keep_count]]

    # Final decision among the remaining candidates using all evaluation points.
    # This gives a stable winner while still saving work in earlier rounds.
    final_scores = consensus_scores(eval_pts, normals[alive], ds[alive], config.threshold)
    best_alive_pos = int(np.argmax(final_scores))
    best_idx = int(alive[best_alive_pos])
    best_n = normals[best_idx]
    best_d = float(ds[best_idx])
    best_score = float(final_scores[best_alive_pos])

    ref_n, ref_d, inliers = refine_plane_iteratively(
        pts, best_n, best_d, config.threshold, config.refine_iterations
    )
    runtime = time.perf_counter() - start
    return PlaneResult(
        normal=ref_n.tolist(),
        d=float(ref_d),
        valid_mask=valid_mask,
        inlier_mask_valid=inliers,
        count=int(inliers.sum()),
        num_valid=int(len(pts)),
        method="preemptive",
        actual_iterations=int(attempts),
        score=float(best_score),
        runtime_s=float(runtime),
    )


def estimate_plane(points: np.ndarray, method: str, config: RansacConfig) -> PlaneResult:
    """Dispatch plane estimation to the selected RANSAC variant."""
    if method == "ransac":
        return ransac_plane_standard(points, config, method_name="ransac")
    if method == "mlesac":
        return ransac_plane_mlesac(points, config)
    if method == "preemptive":
        return ransac_plane_preemptive(points, config)
    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# Mask and box-dimension helpers
# -----------------------------------------------------------------------------

def valid_inlier_mask_to_image(valid_flat: np.ndarray, inlier_valid: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Map inliers from a valid-point list back to image coordinates."""
    H, W = shape_hw
    out = np.zeros(H * W, dtype=bool)
    valid_idx = np.flatnonzero(valid_flat)
    out[valid_idx[inlier_valid]] = True
    return out.reshape(H, W)


def largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return the largest connected component of a binary mask."""
    labels, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, num + 1))
    best_label = int(np.argmax(sizes)) + 1
    return labels == best_label, int(sizes[best_label - 1])


def orthonormal_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u.astype(np.float32), v.astype(np.float32), normal.astype(np.float32)


def oriented_box_dimensions(points: np.ndarray, plane_normal: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute length and width using a PCA-aligned 2D bounding box."""
    if len(points) < 3:
        raise ValueError("Not enough points to estimate box dimensions.")

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

    corners2 = np.array(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float32,
    )

    basis2 = np.stack([u, v], axis=1) @ evecs
    corners3 = centroid + corners2 @ basis2.T

    length = float(dims[0])
    width = float(dims[1])
    if width > length:
        length, width = width, length
    return length, width, corners3.astype(np.float32)


def compute_quality_score(
    normal_parallelism: float,
    height_mean: float,
    height_plane: float,
    top_component_pixels: int,
) -> float:
    """Heuristic quality score for hyperparameter tuning without ground truth.

    The provided exercise data do not include ground-truth dimensions. Therefore,
    the sweep uses internal consistency criteria: parallel floor/top planes,
    plausible height, agreement between two height estimates, and a sufficiently
    large connected top component.
    """
    parallel_score = np.clip((normal_parallelism - 0.90) / 0.10, 0.0, 1.0)
    height_plausible = 1.0 if 0.05 <= height_mean <= 1.50 else 0.0
    height_consistency = math.exp(-abs(height_mean - height_plane) / 0.025)
    component_score = np.clip(top_component_pixels / 12000.0, 0.0, 1.0)
    return float(
        0.35 * parallel_score
        + 0.25 * height_plausible
        + 0.25 * height_consistency
        + 0.15 * component_score
    )


# -----------------------------------------------------------------------------
# Data loading and visualizations
# -----------------------------------------------------------------------------

def load_example(example_idx: int, input_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(input_dir, f"example{example_idx}kinect.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Place the example .mat files inside the data folder."
        )
    data = sio.loadmat(path)
    return (
        data[f"amplitudes{example_idx}"],
        data[f"distances{example_idx}"],
        data[f"cloud{example_idx}"],
    )


def save_input_visualization(example_idx: int, A: np.ndarray, D: np.ndarray, PC: np.ndarray, out_dir: str) -> None:
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
    plt.savefig(os.path.join(out_dir, f"example{example_idx}_inputs.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_detection_visualization(
    example_idx: int,
    method: str,
    D: np.ndarray,
    floor_mask: np.ndarray,
    floor_mask_filtered: np.ndarray,
    top_mask_raw: np.ndarray,
    top_mask_box: np.ndarray,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    valid = ~np.isnan(D)

    axes[0].imshow(np.where(valid, D, np.nan), cmap="viridis")
    axes[0].set_title("distance")
    axes[1].imshow(floor_mask, cmap="gray")
    axes[1].set_title("floor mask raw")
    axes[2].imshow(floor_mask_filtered, cmap="gray")
    axes[2].set_title("floor mask filtered")
    axes[3].imshow(top_mask_raw, cmap="gray")
    axes[3].set_title("top plane mask")

    vis = np.zeros((top_mask_box.shape[0], top_mask_box.shape[1], 3), dtype=np.float32)
    vis[:, :] = [0.0, 0.4, 1.0]          # background
    vis[floor_mask_filtered] = [0.6, 0.9, 0.4]
    vis[top_mask_box] = [0.8, 0.0, 0.0]
    axes[4].imshow(vis)
    axes[4].set_title("largest CC = box top")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Example {example_idx} - {method}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"example{example_idx}_{method}_masks.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_3d_visualization(
    example_idx: int,
    method: str,
    PC: np.ndarray,
    floor_mask_filtered: np.ndarray,
    top_mask_box: np.ndarray,
    corners: np.ndarray,
    out_dir: str,
) -> None:
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
    ax.plot(cyc[:, 0], cyc[:, 1], cyc[:, 2], linewidth=2, label="estimated top outline")

    ax.set_title(f"Example {example_idx}: {method}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example{example_idx}_{method}_3d.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_presentation_visualization(
    example_idx: int,
    method: str,
    PC: np.ndarray,
    floor_mask_filtered: np.ndarray,
    top_mask_box: np.ndarray,
    corners: np.ndarray,
    out_dir: str,
) -> None:
    """Clean 2D visualization with labels similar to the group result."""
    H, W, _ = PC.shape

    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:, :] = np.array([0.00, 0.40, 1.00], dtype=np.float32)      # blue background
    img[floor_mask_filtered] = np.array([0.60, 0.93, 0.45], dtype=np.float32)  # green floor
    img[top_mask_box] = np.array([0.70, 0.00, 0.00], dtype=np.float32)         # red box top

    edge_color = np.array([0.00, 0.95, 1.00], dtype=np.float32)  # cyan outline

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img, origin="upper")

    top_coords = np.argwhere(top_mask_box)
    top_pts = PC[top_mask_box]

    if len(top_pts) > 0:
        corner_pixels = []

        for c in corners:
            distances = np.linalg.norm(top_pts - c[None, :], axis=1)
            idx = int(np.argmin(distances))
            r, col = top_coords[idx]
            corner_pixels.append([col, r])

        corner_pixels = np.array(corner_pixels, dtype=float)
        cyc = np.vstack([corner_pixels, corner_pixels[0]])

        ax.plot(cyc[:, 0], cyc[:, 1], color=edge_color, linewidth=2.5)

        # Labels like the group task
        cx = corner_pixels[:, 0].mean()
        cy = corner_pixels[:, 1].mean()

        ax.text(cx, cy - 35, "top", color="black", fontsize=9, ha="center")
        ax.text(corner_pixels[:, 0].min() - 22, cy, "left", color="black", fontsize=9, ha="center")
        ax.text(corner_pixels[:, 0].max() + 22, cy, "right", color="black", fontsize=9, ha="center")
        ax.text(cx + 10, corner_pixels[:, 1].max() + 16, "bottom", color="black", fontsize=9, ha="center")

    ax.set_title(f"Example {example_idx}: {method} result")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"example{example_idx}_{method}_presentation.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_method_comparison_plot(results: List[DetectionResult], out_dir: str) -> None:
    """Create per-example method comparison plots."""
    by_example: Dict[int, List[DetectionResult]] = {}
    for r in results:
        by_example.setdefault(r.example, []).append(r)

    for example, rows in sorted(by_example.items()):
        methods = [r.method for r in rows]
        heights = [r.height_m_plane_distance for r in rows]
        runtimes = [r.total_runtime_s for r in rows]
        scores = [r.quality_score for r in rows]

        x = np.arange(len(methods))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, heights)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("height estimate (m)")
        ax.set_title(f"Example {example}: height by RANSAC variant")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example{example}_method_height_comparison.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, runtimes)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("runtime (s)")
        ax.set_title(f"Example {example}: runtime by RANSAC variant")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example{example}_method_runtime_comparison.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, scores)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("internal quality score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Example {example}: quality score by RANSAC variant")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example{example}_method_quality_comparison.png"), dpi=180)
        plt.close(fig)


def save_tuning_plot(rows: List[DetectionResult], out_dir: str, example_idx: int, method: str) -> None:
    """Visualize how threshold changes affect dimensions and runtime."""
    rows = sorted(rows, key=lambda r: (r.floor_threshold_m, r.max_iterations))
    thresholds = np.array([r.floor_threshold_m for r in rows])
    iterations = np.array([r.max_iterations for r in rows])
    labels = [f"t={r.floor_threshold_m:.3f}\nN={r.max_iterations}" for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.55), 4.8))
    ax.plot(x, [r.length_m for r in rows], marker="o", label="length")
    ax.plot(x, [r.width_m for r in rows], marker="o", label="width")
    ax.plot(x, [r.height_m_plane_distance for r in rows], marker="o", label="height")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("dimension (m)")
    ax.set_title(f"Example {example_idx}: dimension change during tuning ({method})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example{example_idx}_{method}_tuning_dimensions.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.55), 4.8))
    ax.plot(x, [r.total_runtime_s for r in rows], marker="o", label="runtime")
    ax.plot(x, [r.quality_score for r in rows], marker="o", label="quality score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Example {example_idx}: runtime/quality change during tuning ({method})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example{example_idx}_{method}_tuning_runtime_quality.png"), dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Full detection pipeline
# -----------------------------------------------------------------------------

def solve_example(
    example_idx: int,
    input_dir: str,
    out_dir: str,
    method: str,
    floor_threshold: float,
    top_threshold: float,
    max_iterations: int,
    seed: int,
    save_outputs: bool = True,
) -> DetectionResult:
    """Run the complete floor/box detection pipeline for one example."""
    total_start = time.perf_counter()
    A, D, PC = load_example(example_idx, input_dir)
    H, W, _ = PC.shape

    os.makedirs(out_dir, exist_ok=True)
    if save_outputs and not os.path.exists(os.path.join(out_dir, f"example{example_idx}_inputs.png")):
        save_input_visualization(example_idx, A, D, PC, out_dir)

    floor_config = RansacConfig(
        threshold=floor_threshold,
        max_iterations=max_iterations,
        batch_size=128,
        seed=seed,
        eval_points=6000,
        adaptive=(method == "ransac"),
        preemptive_candidates=max(256, max_iterations * 2),
    )
    floor = estimate_plane(PC.reshape(-1, 3), method, floor_config)

    floor_mask = valid_inlier_mask_to_image(
        floor.valid_mask,
        floor.inlier_mask_valid,
        (H, W),
    )

    # Closing fills small holes; opening removes small isolated blobs.
    floor_mask_filtered = ndimage.binary_closing(floor_mask, structure=np.ones((7, 7)))
    floor_mask_filtered = ndimage.binary_opening(floor_mask_filtered, structure=np.ones((5, 5)))

    non_floor_mask = (~floor_mask_filtered) & (PC[..., 2] != 0)
    non_floor_points = PC[non_floor_mask].reshape(-1, 3)

    top_config = RansacConfig(
        threshold=top_threshold,
        max_iterations=max(80, int(max_iterations * 0.85)),
        batch_size=96,
        seed=seed + 1000,
        eval_points=min(5000, len(non_floor_points)),
        adaptive=(method == "ransac"),
        preemptive_candidates=max(192, int(max_iterations * 1.5)),
    )
    top = estimate_plane(non_floor_points, method, top_config)

    top_local_mask = np.zeros(len(non_floor_points), dtype=bool)
    top_valid_idx = np.flatnonzero(top.valid_mask)
    top_local_mask[top_valid_idx[top.inlier_mask_valid]] = True

    non_floor_coords = np.argwhere(non_floor_mask)
    top_mask_raw = np.zeros((H, W), dtype=bool)
    top_mask_raw[
        non_floor_coords[top_local_mask, 0],
        non_floor_coords[top_local_mask, 1],
    ] = True

    top_mask_raw = ndimage.binary_opening(top_mask_raw, structure=np.ones((3, 3)))
    top_mask_raw = ndimage.binary_closing(top_mask_raw, structure=np.ones((7, 7)))
    top_mask_box, cc_size = largest_connected_component(top_mask_raw)

    if cc_size < 3:
        raise RuntimeError(
            f"Example {example_idx}, method {method}: box top component too small. "
            "Try a larger top threshold or more RANSAC iterations."
        )

    box_points = PC[top_mask_box]
    top_n, top_d = fit_plane_svd(box_points)
    floor_n = np.array(floor.normal, dtype=np.float32)
    floor_d = float(floor.d)

    if np.dot(top_n, floor_n) < 0:
        top_n = -top_n
        top_d = -top_d

    distances_to_floor = np.abs(box_points @ floor_n - floor_d)
    height_mean = float(np.mean(distances_to_floor))
    height_median = float(np.median(distances_to_floor))
    height_planes = float(abs(top_d - floor_d))
    parallelism = float(abs(np.dot(top_n, floor_n)))

    length, width, corners = oriented_box_dimensions(box_points, top_n)
    total_runtime = time.perf_counter() - total_start

    quality = compute_quality_score(
        normal_parallelism=parallelism,
        height_mean=height_mean,
        height_plane=height_planes,
        top_component_pixels=cc_size,
    )

    if save_outputs:
        save_detection_visualization(
            example_idx,
            method,
            D,
            floor_mask,
            floor_mask_filtered,
            top_mask_raw,
            top_mask_box,
            out_dir,
        )
        save_3d_visualization(example_idx, method, PC, floor_mask_filtered, top_mask_box, corners, out_dir)
        save_presentation_visualization(example_idx, method, PC, floor_mask_filtered, top_mask_box, corners, out_dir)

    return DetectionResult(
        example=example_idx,
        method=method,
        seed=seed,
        floor_threshold_m=float(floor_threshold),
        top_threshold_m=float(top_threshold),
        max_iterations=int(max_iterations),
        floor_plane_normal=[float(x) for x in floor.normal],
        floor_plane_d=float(floor.d),
        top_plane_normal=[float(x) for x in top_n.tolist()],
        top_plane_d=float(top_d),
        num_floor_inliers=int(floor.count),
        num_floor_valid_points=int(floor.num_valid),
        num_top_inliers=int(top.count),
        num_top_valid_points=int(top.num_valid),
        top_component_pixels=int(cc_size),
        normal_parallelism_abs_dot=parallelism,
        height_m_mean_point_to_floor=height_mean,
        height_m_median_point_to_floor=height_median,
        height_m_plane_distance=height_planes,
        length_m=float(length),
        width_m=float(width),
        quality_score=float(quality),
        floor_runtime_s=float(floor.runtime_s),
        top_runtime_s=float(top.runtime_s),
        total_runtime_s=float(total_runtime),
        floor_iterations=int(floor.actual_iterations),
        top_iterations=int(top.actual_iterations),
    )


# -----------------------------------------------------------------------------
# Evaluation routines
# -----------------------------------------------------------------------------

def write_results_csv(path: str, results: Sequence[DetectionResult]) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_results_json(path: str, results: Sequence[DetectionResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def run_method_comparison(
    examples: Sequence[int],
    methods: Sequence[str],
    input_dir: str,
    out_dir: str,
    floor_threshold: float,
    top_threshold: float,
    max_iterations: int,
    seed: int,
) -> List[DetectionResult]:
    results: List[DetectionResult] = []
    for example in examples:
        for method in methods:
            print(f"[compare] example={example}, method={method}")
            result = solve_example(
                example,
                input_dir,
                out_dir,
                method,
                floor_threshold,
                top_threshold,
                max_iterations,
                seed + example * 17,
                save_outputs=True,
            )
            results.append(result)
            print(
                f"  L={result.length_m:.3f} m, W={result.width_m:.3f} m, "
                f"H={result.height_m_plane_distance:.3f} m, "
                f"runtime={result.total_runtime_s:.3f}s, quality={result.quality_score:.3f}"
            )

    write_results_csv(os.path.join(out_dir, "individual_method_comparison.csv"), results)
    write_results_json(os.path.join(out_dir, "individual_method_comparison.json"), results)
    save_method_comparison_plot(results, out_dir)
    return results


def run_hyperparameter_tuning(
    examples: Sequence[int],
    methods: Sequence[str],
    input_dir: str,
    out_dir: str,
    thresholds: Sequence[float],
    iterations_grid: Sequence[int],
    seed: int,
) -> List[DetectionResult]:
    results: List[DetectionResult] = []
    for example in examples:
        for method in methods:
            method_rows: List[DetectionResult] = []
            for threshold in thresholds:
                for max_iter in iterations_grid:
                    print(f"[tune] example={example}, method={method}, threshold={threshold}, N={max_iter}")
                    try:
                        result = solve_example(
                            example,
                            input_dir,
                            out_dir,
                            method,
                            floor_threshold=float(threshold),
                            top_threshold=float(threshold),
                            max_iterations=int(max_iter),
                            seed=seed + example * 31 + int(threshold * 10000) + max_iter,
                            save_outputs=False,
                        )
                        results.append(result)
                        method_rows.append(result)
                    except Exception as exc:
                        print(f"  skipped due to error: {exc}")
            if method_rows:
                save_tuning_plot(method_rows, out_dir, example, method)

    write_results_csv(os.path.join(out_dir, "individual_hyperparameter_tuning.csv"), results)
    write_results_json(os.path.join(out_dir, "individual_hyperparameter_tuning.json"), results)

    # Store the best setting per example and method.
    best_rows: List[DetectionResult] = []
    for example in examples:
        for method in methods:
            subset = [r for r in results if r.example == example and r.method == method]
            if subset:
                best_rows.append(max(subset, key=lambda r: r.quality_score))
    write_results_csv(os.path.join(out_dir, "individual_best_hyperparameters.csv"), best_rows)
    return results


def run_timing_analysis(
    examples: Sequence[int],
    methods: Sequence[str],
    input_dir: str,
    out_dir: str,
    floor_threshold: float,
    top_threshold: float,
    max_iterations: int,
    repeats: int,
    seed: int,
) -> List[DetectionResult]:
    rows: List[DetectionResult] = []
    for repeat in range(repeats):
        for example in examples:
            for method in methods:
                print(f"[timing] repeat={repeat + 1}/{repeats}, example={example}, method={method}")
                result = solve_example(
                    example,
                    input_dir,
                    out_dir,
                    method,
                    floor_threshold,
                    top_threshold,
                    max_iterations,
                    seed + repeat * 1000 + example * 41,
                    save_outputs=False,
                )
                rows.append(result)

    write_results_csv(os.path.join(out_dir, "individual_timing_runs.csv"), rows)
    write_results_json(os.path.join(out_dir, "individual_timing_runs.json"), rows)

    summary = []
    for method in methods:
        method_rows = [r for r in rows if r.method == method]
        runtimes = np.array([r.total_runtime_s for r in method_rows], dtype=float)
        summary.append(
            {
                "method": method,
                "runs": int(len(runtimes)),
                "mean_runtime_s": float(np.mean(runtimes)),
                "std_runtime_s": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
                "mean_quality_score": float(np.mean([r.quality_score for r in method_rows])),
                "mean_height_m": float(np.mean([r.height_m_plane_distance for r in method_rows])),
            }
        )

    with open(os.path.join(out_dir, "individual_timing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [s["method"] for s in summary]
    means = [s["mean_runtime_s"] for s in summary]
    ax.bar(np.arange(len(labels)), means)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean runtime (s)")
    ax.set_title("Runtime analysis across examples and repeats")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "individual_timing_summary.png"), dpi=180)
    plt.close(fig)

    run_hypothesis_test(rows, out_dir)
    return rows


def run_hypothesis_test(rows: Sequence[DetectionResult], out_dir: str) -> None:
    """Test whether preemptive RANSAC is faster than standard RANSAC.

    H0: Standard RANSAC and preemptive RANSAC have equal mean runtime.
    H1: Standard RANSAC has larger mean runtime than preemptive RANSAC.
    """
    ransac = [r for r in rows if r.method == "ransac"]
    preemptive = [r for r in rows if r.method == "preemptive"]

    pairs = []
    for r in ransac:
        candidates = [p for p in preemptive if p.example == r.example and p.seed == r.seed]
        if candidates:
            pairs.append((r.total_runtime_s, candidates[0].total_runtime_s))

    result = {
        "hypothesis": "H0: mean runtime(ransac) == mean runtime(preemptive); H1: ransac is slower than preemptive",
        "num_pairs": len(pairs),
        "test": "paired t-test, one-sided where available",
        "p_value": None,
        "mean_ransac_runtime_s": None,
        "mean_preemptive_runtime_s": None,
        "speedup_ransac_over_preemptive": None,
        "interpretation": "Not enough paired runs available.",
    }

    if len(pairs) >= 2:
        arr = np.array(pairs, dtype=float)
        r_times = arr[:, 0]
        p_times = arr[:, 1]
        result["mean_ransac_runtime_s"] = float(np.mean(r_times))
        result["mean_preemptive_runtime_s"] = float(np.mean(p_times))
        result["speedup_ransac_over_preemptive"] = float(np.mean(r_times) / max(np.mean(p_times), 1e-12))

        if stats is not None:
            try:
                test = stats.ttest_rel(r_times, p_times, alternative="greater")
                p_value = float(test.pvalue)
            except TypeError:
                # Older SciPy versions do not support the alternative argument.
                test = stats.ttest_rel(r_times, p_times)
                p_two_sided = float(test.pvalue)
                p_value = p_two_sided / 2.0 if np.mean(r_times - p_times) > 0 else 1.0 - p_two_sided / 2.0
            result["p_value"] = p_value
            result["interpretation"] = (
                "Reject H0 at alpha=0.05: preemptive RANSAC is faster on these runs."
                if p_value < 0.05
                else "Do not reject H0 at alpha=0.05: the measured speed difference is not statistically strong."
            )
        else:
            result["interpretation"] = "SciPy stats was unavailable; only descriptive speedup is reported."

    with open(os.path.join(out_dir, "individual_hypothesis_test_runtime.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Individual Exercise 1: RANSAC variants for box detection")
    parser.add_argument("--input-dir", default="data", help="Folder containing example*kinect.mat files")
    parser.add_argument("--out-dir", default="results_individual", help="Output folder")
    parser.add_argument(
        "--mode",
        choices=["detect", "compare", "tune", "timing", "full"],
        default="full",
        help="Which part of the individual evaluation to run",
    )
    parser.add_argument("--examples", default="1,2,3,4", help="Comma-separated example indices")
    parser.add_argument("--methods", default="ransac,mlesac,preemptive", help="Comma-separated methods")
    parser.add_argument("--floor-threshold", type=float, default=0.025, help="Default floor inlier threshold in metres")
    parser.add_argument("--top-threshold", type=float, default=0.025, help="Default top-plane inlier threshold in metres")
    parser.add_argument("--max-iterations", type=int, default=300, help="RANSAC iteration/candidate budget")
    parser.add_argument("--threshold-grid", default="0.015,0.020,0.025,0.030,0.035", help="Thresholds for tuning")
    parser.add_argument("--iteration-grid", default="150,300,600", help="Iteration budgets for tuning")
    parser.add_argument("--timing-repeats", type=int, default=3, help="Number of repeats for timing analysis")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    examples = parse_int_list(args.examples)
    methods = parse_str_list(args.methods)
    thresholds = parse_float_list(args.threshold_grid)
    iteration_grid = parse_int_list(args.iteration_grid)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode in {"detect", "compare", "full"}:
        run_method_comparison(
            examples=examples,
            methods=methods,
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            floor_threshold=args.floor_threshold,
            top_threshold=args.top_threshold,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )

    if args.mode in {"tune", "full"}:
        run_hyperparameter_tuning(
            examples=examples,
            methods=methods,
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            thresholds=thresholds,
            iterations_grid=iteration_grid,
            seed=args.seed,
        )

    if args.mode in {"timing", "full"}:
        run_timing_analysis(
            examples=examples,
            methods=methods,
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            floor_threshold=args.floor_threshold,
            top_threshold=args.top_threshold,
            max_iterations=args.max_iterations,
            repeats=args.timing_repeats,
            seed=args.seed,
        )

    print("\nFinished individual evaluation.")
    print(f"Results were written to: {args.out_dir}")


if __name__ == "__main__":
    main()
