from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
JPG_DIR = DATA_DIR / "hdr-jpg"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

JPEG_QUALITY = 99
EPS = 1e-8


def read_jpg(path):
    img = imageio.imread(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def save_jpg(path, img, quality=JPEG_QUALITY):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)

    img_uint8 = (img * 255.0 + 0.5).astype(np.uint8)
    imageio.imwrite(path, img_uint8, quality=quality)


def luminance(img):
    return (
        0.299 * img[:, :, 0]
        + 0.587 * img[:, :, 1]
        + 0.114 * img[:, :, 2]
    )


def percentile_normalize(data, low=0.01, high=99.90):
    data = np.nan_to_num(
        data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    a = np.percentile(data, low)
    b = np.percentile(data, high)

    if b <= a:
        return np.zeros_like(data, dtype=np.float32)

    out = (data - a) / (b - a)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_jpg_sequence():
    jpg_files = sorted(
        [p for p in JPG_DIR.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg"]]
    )

    if len(jpg_files) == 0:
        print("[Exercise 9] No JPG files found.")
        print("Expected folder: data/hdr-jpg/")
        return [], []

    jpg_images = [read_jpg(path) for path in jpg_files]

    brightness = [float(np.mean(luminance(img))) for img in jpg_images]
    order = np.argsort(brightness)[::-1]

    jpg_files = [jpg_files[i] for i in order]
    jpg_images = [jpg_images[i] for i in order]
    brightness = [brightness[i] for i in order]

    print("[Exercise 9] Files sorted from brightest to darkest:")
    for file, b in zip(jpg_files, brightness):
        print(f"  {file.name:15s} mean brightness={b:.4f}")

    return jpg_files, jpg_images


def estimate_doubling_map(
    jpg_images,
    bins=180,
    sample_step=12,
    min_value=0.025,
    max_value=0.92,
    min_samples_per_bin=25,
):
    dark_values = []
    bright_values = []

    for i in range(len(jpg_images) - 1):
        y_bright = luminance(jpg_images[i])[::sample_step, ::sample_step]
        y_dark = luminance(jpg_images[i + 1])[::sample_step, ::sample_step]

        valid = (
            (y_dark > min_value)
            & (y_dark < max_value)
            & (y_bright > min_value)
            & (y_bright < max_value)
        )

        if np.any(valid):
            dark_values.append(y_dark[valid].ravel())
            bright_values.append(y_bright[valid].ravel())

    if len(dark_values) == 0:
        raise RuntimeError("Not enough valid JPG pixel pairs to estimate the curve.")

    dark_values = np.concatenate(dark_values)
    bright_values = np.concatenate(bright_values)

    edges = np.linspace(min_value, max_value, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mapped = np.full(bins, np.nan, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int32)

    bin_ids = np.digitize(dark_values, edges) - 1

    for b in range(bins):
        sel = bin_ids == b
        counts[b] = int(np.sum(sel))
        if counts[b] >= min_samples_per_bin:
            mapped[b] = float(np.median(bright_values[sel]))

    valid_bins = np.isfinite(mapped)
    if np.sum(valid_bins) < 8:
        raise RuntimeError(
            "Too few populated intensity bins to estimate an empirical response curve."
        )

    mapped_filled = np.interp(
        centers,
        centers[valid_bins],
        mapped[valid_bins],
    )

    mapped_filled = np.maximum.accumulate(mapped_filled)
    mapped_filled = np.clip(mapped_filled, min_value, max_value)

    mapped_filled = np.maximum(mapped_filled, centers + 1e-4)
    mapped_filled = np.clip(mapped_filled, min_value, max_value)

    print(
        f"[Exercise 9] Empirical doubling map estimated from "
        f"{dark_values.size} paired luminance samples."
    )

    return centers.astype(np.float32), mapped_filled.astype(np.float32), counts


def build_inverse_curve_from_doublings(
    map_input,
    map_doubled,
    start_value=0.045,
    stop_value=0.88,
    max_steps=12,
):
    lo = float(map_input[0])
    hi = float(map_input[-1])
    y = float(np.clip(start_value, lo, hi))

    y_anchors = [y]
    x_anchors = [1.0]

    for _ in range(max_steps):
        y_next = float(np.interp(y, map_input, map_doubled))

        if y_next <= y + 0.003:
            break

        y_anchors.append(y_next)
        x_anchors.append(x_anchors[-1] * 2.0)
        y = y_next

        if y >= stop_value:
            break

    y_anchors = np.asarray(y_anchors, dtype=np.float64)
    x_anchors = np.asarray(x_anchors, dtype=np.float64)

    if len(y_anchors) < 4:
        raise RuntimeError(
            "The empirical doubling map did not produce enough response anchors."
        )

    keep = np.concatenate(([True], np.diff(y_anchors) > 1e-4))
    y_anchors = y_anchors[keep]
    x_anchors = x_anchors[keep]

    log_x = np.log(x_anchors + EPS)

    first_slope = (log_x[1] - log_x[0]) / (y_anchors[1] - y_anchors[0] + EPS)
    last_slope = (log_x[-1] - log_x[-2]) / (
        y_anchors[-1] - y_anchors[-2] + EPS
    )

    y_curve = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    log_curve = np.interp(y_curve, y_anchors, log_x)

    below = y_curve < y_anchors[0]
    above = y_curve > y_anchors[-1]

    log_curve[below] = log_x[0] + first_slope * (y_curve[below] - y_anchors[0])
    log_curve[above] = log_x[-1] + last_slope * (y_curve[above] - y_anchors[-1])

    linear_curve = np.exp(log_curve)

    linear_curve = linear_curve - linear_curve[0]
    linear_curve = np.maximum(linear_curve, 0.0)

    ref_idx = int(0.5 * (len(linear_curve) - 1))
    ref = linear_curve[ref_idx]
    if ref <= EPS:
        ref = np.max(linear_curve)
    linear_curve = linear_curve / (ref + EPS)

    linear_curve = np.maximum.accumulate(linear_curve)

    print("[Exercise 9] Inverse JPG curve built from exposure doublings.")
    print("[Exercise 9] Curve anchors (JPG -> relative linear light):")
    for yy, xx in zip(y_anchors, x_anchors):
        print(f"  {yy:.4f} -> {xx:.1f}")

    return (
        y_curve.astype(np.float32),
        linear_curve.astype(np.float32),
        y_anchors.astype(np.float32),
        x_anchors.astype(np.float32),
    )


def save_curve_plot(map_input, map_doubled, curve_y, curve_x, y_anchors, x_anchors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(map_input, map_doubled, color="tab:blue", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("JPG value at shorter exposure")
    axes[0].set_ylabel("JPG value after 2x exposure")
    axes[0].set_title("Empirical exposure-doubling map")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(curve_y, curve_x, color="tab:orange", linewidth=2)
    anchor_scaled = np.interp(y_anchors, curve_y, curve_x)
    axes[1].plot(y_anchors, anchor_scaled, "o", color="black", markersize=4)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Observed JPG value")
    axes[1].set_ylabel("Estimated relative linear light")
    axes[1].set_title("Estimated inverse camera curve")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exercise 9: Empirical Camera Curve from Exposure Doublings")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "exercise9_empirical_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[Exercise 9] Saved results/exercise9_empirical_curve.png")


def linearize_jpg(img, curve_y, curve_x):
    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(
            img[:, :, c], curve_y, curve_x
        ).astype(np.float32)
    return out


def merge_jpg_hdr_by_best_exposure(
    jpg_images,
    exposure_times,
    curve_y,
    curve_x,
    saturation_limit=0.88,
):
    h, w, c = jpg_images[0].shape
    hdr = np.zeros((h, w, c), dtype=np.float32)
    chosen = np.zeros((h, w), dtype=bool)

    for i, img in enumerate(jpg_images):
        linear_img = linearize_jpg(img, curve_y, curve_x)

        radiance = linear_img / float(exposure_times[i])

        if i < len(jpg_images) - 1:
            not_saturated = np.all(img < saturation_limit, axis=2)
            take = (~chosen) & not_saturated
        else:
            take = ~chosen

        hdr[take] = radiance[take]
        chosen[take] = True

        print(
            f"[Exercise 9] image {i:02d}: "
            f"used {np.mean(take) * 100:.2f}% pixels"
        )

    print(f"[Exercise 9] unchosen pixels: {np.mean(~chosen) * 100:.4f}%")
    return hdr.astype(np.float32)


def log_tone_mapping(hdr):
    hdr = np.maximum(hdr.astype(np.float32), 0.0)
    log_hdr = np.log1p(hdr)
    return percentile_normalize(log_hdr, 0.01, 99.90)


def final_display_adjustment(img, contrast=1.25, saturation=1.20, brightness=1.04):
    img = np.clip(img.astype(np.float32), 0.0, 1.0)

    img = np.clip(img * brightness, 0.0, 1.0)
    img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)

    gray = np.mean(img, axis=2, keepdims=True)
    img = gray + saturation * (img - gray)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def exercise9_jpg_hdr():
    jpg_files, jpg_images = load_jpg_sequence()

    if len(jpg_images) == 0:
        return

    if len(jpg_images) < 2:
        print("[Exercise 9] Need at least two differently exposed JPG images.")
        return

    print(f"[Exercise 9] Number of JPG files: {len(jpg_images)}")

    exposure_times = np.array(
        [
        13.0,
        6.0,
        3.2,
        1.6,
        0.8,
        0.4,
        1/5,
        1/10,
        1/20,
        1/40,
        1/80,
        1/160,
    ],
        dtype=np.float32,
    )

    map_input, map_doubled, _ = estimate_doubling_map(jpg_images[2:])
    curve_y, curve_x, y_anchors, x_anchors = build_inverse_curve_from_doublings(
        map_input,
        map_doubled,
    )

    save_curve_plot(
        map_input,
        map_doubled,
        curve_y,
        curve_x,
        y_anchors,
        x_anchors,
    )

    hdr = merge_jpg_hdr_by_best_exposure(
        jpg_images,
        exposure_times,
        curve_y=curve_y,
        curve_x=curve_x,
        saturation_limit=0.88,
    )

    hdr_log = log_tone_mapping(hdr)

    save_jpg(
        RESULTS_DIR / "exercise9_jpg_hdr_log_raw.jpg",
        hdr_log,
        quality=JPEG_QUALITY,
    )
    print("[Exercise 9] Saved results/exercise9_jpg_hdr_log_raw.jpg")

    hdr_final = final_display_adjustment(
        hdr_log,
        contrast=1.25,
        saturation=1.20,
        brightness=1.04,
    )

    save_jpg(
        RESULTS_DIR / "exercise9_jpg_hdr_result.jpg",
        hdr_final,
        quality=JPEG_QUALITY,
    )
    print("[Exercise 9] Saved results/exercise9_jpg_hdr_result.jpg")
    print("[Exercise 9] Done.")


if __name__ == "__main__":
    exercise9_jpg_hdr()