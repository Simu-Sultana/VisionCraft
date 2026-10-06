from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage import convolve

try:
    from skimage.restoration import denoise_bilateral
except Exception:  # fallback if scikit-image is unavailable
    denoise_bilateral = None
    from scipy.ndimage import gaussian_filter


# ============================================================
# Paths and constants
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

JPEG_QUALITY = 99
EPS = 1e-8

# Exercise 1 result for IMG_9939.npy.
# From manual inspection of the red/green/blue pen image:
#   G B
#   R G
EXERCISE1_PATTERN = "GBRG"


# ============================================================
# Loading RAW data and Bayer masks
# ============================================================

def load_raw_array(path: os.PathLike | str) -> np.ndarray:
    """Load visible raw sensor data from a CR3 file as float32."""
    with rawpy.imread(str(path)) as raw:
        raw_array = np.array(raw.raw_image_visible, dtype=np.float32)
    return raw_array


def load_raw_array_and_masks(path: os.PathLike | str, subtract_black: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
  
    with rawpy.imread(str(path)) as raw:
        raw_array = np.array(raw.raw_image_visible, dtype=np.float32)
        raw_colors = np.array(raw.raw_colors_visible)

        color_desc = raw.color_desc
        if isinstance(color_desc, bytes):
            color_desc = color_desc.decode("ascii", errors="ignore")
        else:
            color_desc = str(color_desc)

        black_levels = list(getattr(raw, "black_level_per_channel", []) or [])

    masks = {
        "R": np.zeros_like(raw_array, dtype=np.float32),
        "G": np.zeros_like(raw_array, dtype=np.float32),
        "B": np.zeros_like(raw_array, dtype=np.float32),
    }

    # Subtract the per-color black level before any demosaicing.
    if subtract_black and black_levels:
        black = np.zeros_like(raw_array, dtype=np.float32)
        for idx in range(min(len(black_levels), len(color_desc))):
            black[raw_colors == idx] = float(black_levels[idx])
        raw_array = raw_array - black
        raw_array = np.clip(raw_array, 0.0, None)

    for idx, color_char in enumerate(color_desc):
        color_char = color_char.upper()
        if color_char in masks:
            masks[color_char][raw_colors == idx] = 1.0

    total_mask = masks["R"] + masks["G"] + masks["B"]
    if np.sum(total_mask) == 0:
        raise RuntimeError(f"Could not build Bayer masks for {path}. color_desc={color_desc!r}")

    return raw_array.astype(np.float32), masks


def masks_from_pattern(shape: Tuple[int, int], pattern: str) -> Dict[str, np.ndarray]:
    """
    Create Bayer masks for a named 2x2 pattern.

    Pattern names use row-major order:
        RGGB = R G / G B
        BGGR = B G / G R
        GRBG = G R / B G
        GBRG = G B / R G
    """
    h, w = shape
    pattern = pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    masks = {
        "R": np.zeros((h, w), dtype=np.float32),
        "G": np.zeros((h, w), dtype=np.float32),
        "B": np.zeros((h, w), dtype=np.float32),
    }

    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_offset, col_offset, color in positions:
        masks[color][row_offset::2, col_offset::2] = 1.0

    return masks


# ============================================================
# Exercise 2: Demosaicing
# ============================================================

def demosaic_with_masks(raw_data: np.ndarray, masks: Dict[str, np.ndarray], kernel_size: int = 3) -> np.ndarray:
    """
    Simple lecture demosaicing:

        C = ((Mc * X) convolved K) / (Mc convolved K)

    Mc = mask of one color channel
    X  = raw sensor data
    K  = convolution kernel
    """
    raw_data = raw_data.astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    channels = []
    for color in ["R", "G", "B"]:
        mask = masks[color].astype(np.float32)
        numerator = convolve(raw_data * mask, kernel, mode="mirror")
        denominator = convolve(mask, kernel, mode="mirror")
        channel = numerator / np.maximum(denominator, EPS)
        channels.append(channel.astype(np.float32))

    return np.stack(channels, axis=-1).astype(np.float32)


def demosaic_cr3(path: os.PathLike | str, kernel_size: int = 3) -> np.ndarray:
    """Load a CR3 file and demosaic it using masks read from rawpy."""
    raw_data, masks = load_raw_array_and_masks(path)
    return demosaic_with_masks(raw_data, masks, kernel_size=kernel_size)


# ============================================================
# Normalization and saving
# ============================================================

def percentile_bounds(data: np.ndarray, low: float = 0.01, high: float = 99.99) -> Tuple[float, float]:
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a = float(np.percentile(data, low))
    b = float(np.percentile(data, high))
    return a, b


def normalize_with_bounds(data: np.ndarray, a: float, b: float) -> np.ndarray:
    data = data.astype(np.float32)
    if b <= a:
        return np.zeros_like(data, dtype=np.float32)
    out = (data - a) / (b - a)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def percentile_normalize(data: np.ndarray, low: float = 0.01, high: float = 99.99) -> np.ndarray:
    a, b = percentile_bounds(data, low=low, high=high)
    return normalize_with_bounds(data, a, b)




def final_contrast_adjustment(rgb_0_1: np.ndarray, contrast: float = 1.18, gamma: float = 0.95, saturation: float = 1.08) -> np.ndarray:
    
    rgb = np.clip(rgb_0_1.astype(np.float32), 0.0, 1.0)

    # Mild contrast around middle gray.
    rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)

    # Slight brightness lift. Gamma < 1 brightens mid/dark values a bit.
    rgb = np.power(rgb, gamma)

    # Mild saturation correction: move colors slightly away from grayscale.
    gray = np.mean(rgb, axis=2, keepdims=True)
    rgb = gray + saturation * (rgb - gray)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def save_jpg(path: os.PathLike | str, rgb_0_1: np.ndarray, quality: int = JPEG_QUALITY) -> None:
    """Save an RGB image in [0,1] as high-quality JPG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb_0_1 = np.nan_to_num(rgb_0_1, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_0_1 = np.clip(rgb_0_1, 0.0, 1.0)
    rgb_uint8 = (rgb_0_1 * 255.0 + 0.5).astype(np.uint8)

    imageio.imwrite(path, rgb_uint8, quality=quality)


def save_png(path: os.PathLike | str, rgb_0_1: np.ndarray) -> None:
    """Save an RGB image in [0,1] as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_0_1 = np.clip(np.nan_to_num(rgb_0_1), 0.0, 1.0)
    imageio.imwrite(path, (rgb_0_1 * 255.0 + 0.5).astype(np.uint8))


# ============================================================
# Exercise 3: Luminosity curves
# ============================================================

def gamma_correction(data: np.ndarray, gamma: float = 0.3) -> np.ndarray:

    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)
    x = normalize_with_bounds(data, a, b)
    y = np.power(x, gamma)
    return (y * (b - a) + a).astype(np.float32)


def logarithmic_curve(data: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Alternative curve for Exercise 3: log(1 + c*x) / log(1+c)."""
    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)
    x = normalize_with_bounds(data, a, b)
    y = np.log1p(strength * x) / np.log1p(strength)
    return (y * (b - a) + a).astype(np.float32)


def sqrt_curve(data: np.ndarray) -> np.ndarray:
    """Another simple curve: sqrt(x). Softer than gamma=0.3."""
    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)
    x = normalize_with_bounds(data, a, b)
    y = np.sqrt(x)
    return (y * (b - a) + a).astype(np.float32)


def inverse_curve(data: np.ndarray) -> np.ndarray:
    """
    Experimental curve y = 1 - x.

    This is intentionally kept for discussion: it produces a negative-like
    image because dark values become bright and bright values become dark.
    """
    data = data.astype(np.float32)
    a, b = percentile_bounds(data, 0.01, 99.99)
    x = normalize_with_bounds(data, a, b)
    y = 1.0 - x
    return (y * (b - a) + a).astype(np.float32)


# ============================================================
# Exercise 4: Gray-world white balance
# ============================================================

def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    """
    Lecture gray-world method:
        mean value of full image: mi
        mean value of channel c: mc
        channel c <- channel c * mi / mc
    """
    rgb = rgb.astype(np.float32).copy()

    mean_image = float(np.mean(rgb))
    mean_channels = np.mean(rgb, axis=(0, 1))
    scale = mean_image / np.maximum(mean_channels, EPS)

    balanced = rgb * scale.reshape(1, 1, 3)
    balanced = np.nan_to_num(balanced, nan=0.0, posinf=0.0, neginf=0.0)
    balanced = np.clip(balanced, 0.0, None)
    return balanced.astype(np.float32)


# ============================================================
# Exercise 5: Sensor linearity
# ============================================================

def exercise5_sensor_linearity(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> None:
    files = [
        "IMG_3044.CR3",
        "IMG_3045.CR3",
        "IMG_3046.CR3",
        "IMG_3047.CR3",
        "IMG_3048.CR3",
        "IMG_3049.CR3",
    ]

    exposure_times = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160, 1 / 320], dtype=np.float32)
    mean_values: List[float] = []

    for filename in files:
        path = data_dir / filename
        if not path.exists():
            print(f"[Exercise 5] Missing file: {path}")
            return
        raw_data = load_raw_array(path)
        mean_values.append(float(np.mean(raw_data)))  # full raw mean, not channel-wise

    mean_values_np = np.array(mean_values, dtype=np.float32)

    # Linear fit for clearer presentation.
    m, c = np.polyfit(exposure_times, mean_values_np, 1)
    fitted = m * exposure_times + c

    plt.figure(figsize=(10, 5))
    plt.plot(exposure_times, mean_values_np, "o", label="Measured raw mean")
    plt.plot(exposure_times, fitted, "-", label="Linear fit")
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Mean Sensor Value")
    plt.title("Sensor Linearity")
    plt.grid(True)
    plt.legend()
    plt.xticks(exposure_times, ["1/10", "1/20", "1/40", "1/80", "1/160", "1/320"])
    plt.savefig(results_dir / "exercise5_sensor_linearity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("[Exercise 5] Saved results/exercise5_sensor_linearity.png")
    print("[Exercise 5] Mean values:")
    for f, t, v in zip(files, exposure_times, mean_values_np):
        print(f"  {f:12s} exposure={t:.6f}s mean={v:.2f}")


# ============================================================
# Exercise 6: HDR raw combination and log tone mapping
# ============================================================

def combine_hdr_raw(raw_images: List[np.ndarray], exposure_times: np.ndarray, threshold_ratio: float = 0.8) -> np.ndarray:

    if len(raw_images) == 0:
        raise ValueError("No raw images provided for HDR combination.")

    hdr = raw_images[0].astype(np.float32).copy()
    first_exposure = float(exposure_times[0])
    # Slightly conservative threshold: still the lecture method, but avoids
    # saturated plateaus earlier than 0.8 for this data set.
    threshold = threshold_ratio * float(np.max(raw_images[0]))

    for idx in range(1, len(raw_images)):
        current = raw_images[idx].astype(np.float32)
        scale = first_exposure / float(exposure_times[idx])
        current_scaled = current * scale

        replace_mask = hdr > threshold
        hdr[replace_mask] = current_scaled[replace_mask]

        print(f"[Exercise 6] combined image {idx:02d} with scale={scale:g}, replaced={np.mean(replace_mask) * 100:.2f}% pixels")

    return hdr.astype(np.float32)


def log_tone_mapping(rgb: np.ndarray) -> np.ndarray:
    """Apply log scale and normalize result to [0,1]."""
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    log_rgb = np.log1p(rgb)
    return percentile_normalize(log_rgb, 0.01, 99.99)


def exercise6_hdr(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> Tuple[np.ndarray | None, Dict[str, np.ndarray] | None]:
    hdr_files = [f"{i:02d}.CR3" for i in range(11)]
    exposure_times = np.array([1.0 / (2 ** i) for i in range(11)], dtype=np.float32)

    raw_images: List[np.ndarray] = []
    used_exposures: List[float] = []
    first_masks: Dict[str, np.ndarray] | None = None

    for filename, exposure in zip(hdr_files, exposure_times):
        path = data_dir / filename
        if not path.exists():
            print(f"[Exercise 6] Missing file: {path}")
            continue

        raw_data, masks = load_raw_array_and_masks(path)
        if first_masks is None:
            first_masks = masks
        raw_images.append(raw_data)
        used_exposures.append(float(exposure))

    if len(raw_images) == 0 or first_masks is None:
        print("[Exercise 6] No HDR files found.")
        return None, None

    used_exposures_np = np.array(used_exposures, dtype=np.float32)
    hdr_raw = combine_hdr_raw(raw_images, used_exposures_np, threshold_ratio=0.65)

    hdr_rgb = demosaic_with_masks(hdr_raw, first_masks, kernel_size=3)
    hdr_rgb_wb = gray_world_white_balance(hdr_rgb)
    hdr_log = log_tone_mapping(hdr_rgb_wb)

    save_jpg(results_dir / "exercise6_hdr_log.jpg", hdr_log)
    print("[Exercise 6] Saved results/exercise6_hdr_log.jpg")

    return hdr_rgb_wb, first_masks


# ============================================================
# Exercise 7: iCAM06 tone mapping
# ============================================================

def bilateral_filter_gray(image: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
    if denoise_bilateral is not None:
        return denoise_bilateral(
            image,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=None,
        ).astype(np.float32)

    # Fallback: not true bilateral, but keeps the script runnable if skimage is missing.
    return gaussian_filter(image, sigma=sigma_spatial).astype(np.float32)


def icam06_tone_mapping(rgb: np.ndarray, output_range: float = 4.0, sigma_color: float = 0.1, sigma_spatial: float = 2.0) -> np.ndarray:
    """iCAM06-inspired method following the lecture pseudocode."""
    rgb = np.maximum(rgb.astype(np.float32), EPS)

    input_intensity = (20.0 * rgb[:, :, 0] + 40.0 * rgb[:, :, 1] + rgb[:, :, 2]) / 61.0
    input_intensity = np.maximum(input_intensity, EPS)

    r_ratio = rgb[:, :, 0] / input_intensity
    g_ratio = rgb[:, :, 1] / input_intensity
    b_ratio = rgb[:, :, 2] / input_intensity

    log_intensity = np.log(input_intensity)
    log_base = bilateral_filter_gray(log_intensity, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    log_details = log_intensity - log_base

    base_range = float(np.max(log_base) - np.min(log_base))
    if base_range <= EPS:
        return percentile_normalize(rgb, 0.01, 99.99)

    compression = np.log(output_range) / base_range
    log_offset = -float(np.max(log_base)) * compression

    output_intensity = np.exp(log_base * compression + log_offset + log_details)

    out_rgb = np.stack(
        [r_ratio * output_intensity, g_ratio * output_intensity, b_ratio * output_intensity],
        axis=-1,
    )

    return percentile_normalize(out_rgb, 0.01, 99.99)


def exercise7_icam06(hdr_rgb_wb: np.ndarray | None, results_dir: Path = RESULTS_DIR) -> None:
    if hdr_rgb_wb is None:
        print("[Exercise 7] Skipped because HDR RGB image is unavailable.")
        return

    # Small spatial sigma keeps runtime acceptable. These settings can be adjusted visually.
    hdr_icam = icam06_tone_mapping(hdr_rgb_wb, output_range=4.0, sigma_color=0.1, sigma_spatial=2.0)
    save_jpg(results_dir / "exercise7_hdr_icam06.jpg", hdr_icam)
    print("[Exercise 7] Saved results/exercise7_hdr_icam06.jpg")


# ============================================================
# Exercise 8: final process_raw function
# ============================================================

def process_raw(input_path: os.PathLike | str, output_path: os.PathLike | str) -> None:

    rgb = demosaic_cr3(input_path, kernel_size=3)
    rgb = gamma_correction(rgb, gamma=0.3)
    rgb = gray_world_white_balance(rgb)
    rgb_out = percentile_normalize(rgb, 0.01, 99.99)
    rgb_out = final_contrast_adjustment(rgb_out)
    save_jpg(output_path, rgb_out, quality=JPEG_QUALITY)
    print(f"[Exercise 8] Saved {output_path}")


# ============================================================
# Exercise 1 helper: Bayer pattern visualization for IMG_9939.npy
# ============================================================

def exercise1_bayer_patterns(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> None:
    path = data_dir / "IMG_9939.npy"
    if not path.exists():
        print(f"[Exercise 1] Missing file: {path}")
        return

    raw = np.load(path).astype(np.float32)
    for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
        masks = masks_from_pattern(raw.shape, pattern)
        rgb = demosaic_with_masks(raw, masks, kernel_size=3)
        rgb_out = percentile_normalize(rgb, 0.01, 99.99)
        save_png(results_dir / f"exercise1_pattern_{pattern}.png", rgb_out)

    print("[Exercise 1] Saved Bayer pattern comparison images.")
    print(f"[Exercise 1] Selected pattern for IMG_9939.npy: {EXERCISE1_PATTERN} = G B / R G")


# ============================================================
# Exercise 2, 3, 4 pipeline for IMG_4782.CR3
# ============================================================

def exercises2_3_4(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> None:
    raw_path = data_dir / "IMG_4782.CR3"
    if not raw_path.exists():
        print(f"[Exercises 2-4] Missing file: {raw_path}")
        return

    raw_data, masks = load_raw_array_and_masks(raw_path)

    # Exercise 2: simple demosaicing only.
    rgb_demosaic = demosaic_with_masks(raw_data, masks, kernel_size=3)
    save_jpg(results_dir / "exercise2_demosaic.jpg", percentile_normalize(rgb_demosaic, 0.01, 99.99))
    print("[Exercise 2] Saved results/exercise2_demosaic.jpg")

    # Exercise 3: gamma + alternative curves.
    curves = {
        "gamma_0_3": gamma_correction(rgb_demosaic, gamma=0.3),
        "log_curve": logarithmic_curve(rgb_demosaic, strength=5.0),
        "sqrt_curve": sqrt_curve(rgb_demosaic),
        "inverse_1_minus_x": inverse_curve(rgb_demosaic),
    }

    for name, rgb_curve in curves.items():
        save_jpg(results_dir / f"exercise3_{name}.jpg", percentile_normalize(rgb_curve, 0.01, 99.99))
        print(f"[Exercise 3] Saved results/exercise3_{name}.jpg")

    # Exercise 4: apply white balance to each Exercise 3 result.
    for name, rgb_curve in curves.items():
        wb = gray_world_white_balance(rgb_curve)
        save_jpg(results_dir / f"exercise4_wb_after_{name}.jpg", percentile_normalize(wb, 0.01, 99.99))
        print(f"[Exercise 4] Saved results/exercise4_wb_after_{name}.jpg")

    # Final process_raw output on same image.
    process_raw(raw_path, results_dir / "final_result_process_raw.jpg")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Starting corrected Demosaicing & HDR solution...")
    print(f"Data folder:    {DATA_DIR.resolve()}")
    print(f"Results folder: {RESULTS_DIR.resolve()}")

    exercise1_bayer_patterns(DATA_DIR, RESULTS_DIR)
    exercises2_3_4(DATA_DIR, RESULTS_DIR)
    exercise5_sensor_linearity(DATA_DIR, RESULTS_DIR)
    hdr_rgb_wb, _ = exercise6_hdr(DATA_DIR, RESULTSS_DIR) if False else exercise6_hdr(DATA_DIR, RESULTS_DIR)
    exercise7_icam06(hdr_rgb_wb, RESULTS_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
