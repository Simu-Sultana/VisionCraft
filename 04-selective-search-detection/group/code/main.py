'''
Run selective search on the humanities data folders and save visual results.
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import argparse
import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform

from selective_search import selective_search


def resize_if_needed(image, max_side):
    """Resize large images for faster selective search while keeping aspect ratio."""
    if max_side is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = skimage.transform.resize(
        image,
        (new_h, new_w),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(image.dtype)
    return resized


def filter_regions(regions, image_shape, min_area=1200, max_aspect_ratio=3.0, max_boxes=80):
    """Remove duplicate, tiny, very large and very elongated proposals."""
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w

    candidates = []
    seen_rects = set()
    for r in regions:
        rect = r["rect"]
        if rect in seen_rects:
            continue
        seen_rects.add(rect)

        x, y, w, h = rect
        if w <= 0 or h <= 0:
            continue
        if r["size"] < min_area:
            continue
        if w * h > 0.95 * image_area:
            continue

        aspect = max(w / float(h), h / float(w))
        if aspect > max_aspect_ratio:
            continue

        candidates.append(r)

    candidates.sort(key=lambda item: item["size"], reverse=True)
    return candidates[:max_boxes]


def process_image(image_path, output_dir, scale, sigma, min_size, min_area, max_aspect_ratio, max_boxes, max_side):
    image = skimage.io.imread(image_path)
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    image_for_search = resize_if_needed(image, max_side)
    print("Processing:", image_path, "original:", image.shape, "used:", image_for_search.shape)

    _, regions = selective_search(image_for_search, scale=scale, sigma=sigma, min_size=min_size)
    candidates = filter_regions(
        regions,
        image_for_search.shape,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        max_boxes=max_boxes,
    )

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(image_for_search)
    for r in candidates:
        x, y, w, h = r["rect"]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=1
        )
        ax.add_patch(rect)

    ax.set_title(f"{os.path.basename(image_path)}: {len(candidates)} proposals")
    plt.axis("off")

    rel_folder = os.path.basename(os.path.dirname(image_path))
    save_dir = os.path.join(output_dir, rel_folder)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)
    print("Saved:", out_path)
    return out_path, len(regions), len(candidates)


def main():
    parser = argparse.ArgumentParser(description="Selective search for humanities images")
    parser.add_argument("--data", default="data", help="Path to the data folder")
    parser.add_argument("--results", default="results", help="Path to the output folder")
    parser.add_argument("--scale", type=float, default=1000, help="Felzenszwalb scale/k parameter")
    parser.add_argument("--sigma", type=float, default=0.8, help="Felzenszwalb Gaussian sigma")
    parser.add_argument("--min-size", type=int, default=40, help="Felzenszwalb minimum component size")
    parser.add_argument("--min-area", type=int, default=1200, help="Smallest region size to draw")
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0, help="Largest allowed width/height or height/width ratio")
    parser.add_argument("--max-boxes", type=int, default=80, help="Maximum boxes drawn per image")
    parser.add_argument("--max-side", type=int, default=500, help="Resize images so longest side is at most this value; set 0 to disable")
    args = parser.parse_args()

    image_paths = []
    for folder in ["chrisarch", "arthist", "classarch"]:
        image_paths.extend(sorted(glob.glob(os.path.join(args.data, folder, "*.jpg"))))

    if not image_paths:
        raise FileNotFoundError("No images found. Run this script from the exercise-5 root folder or set --data.")

    os.makedirs(args.results, exist_ok=True)
    summary = []
    for image_path in image_paths:
        summary.append(
            process_image(
                image_path,
                args.results,
                args.scale,
                args.sigma,
                args.min_size,
                args.min_area,
                args.max_aspect_ratio,
                args.max_boxes,
                args.max_side,
            )
        )

    print("\nSummary")
    for out_path, n_regions, n_candidates in summary:
        print(f"{out_path}: {n_regions} total regions, {n_candidates} drawn boxes")


if __name__ == "__main__":
    main()
