from __future__ import division, print_function

import argparse
import json
import os
import random

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.feature
import skimage.io
import skimage.measure
import skimage.transform
import skimage.util
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from selective_search import selective_search


HOG_LENGTH = 1764
COLOUR_HIST_LENGTH = 48
EXTRA_FEATURE_LENGTH = 11
FEATURE_LENGTH = HOG_LENGTH + COLOUR_HIST_LENGTH + EXTRA_FEATURE_LENGTH


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(current)
        remaining = []
        for idx in order[1:]:
            if iou(boxes[current], boxes[idx]) <= threshold:
                remaining.append(idx)
        order = np.array(remaining)
    return keep


def parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_coco(split_dir):
    annotation_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(annotation_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {
        image["id"]: {
            "id": image["id"],
            "file_name": image["file_name"],
            "path": os.path.join(split_dir, image["file_name"]),
            "boxes": [],
        }
        for image in coco["images"]
    }

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        if image_id in images:
            images[image_id]["boxes"].append(tuple(annotation["bbox"]))

    return list(images.values())


def filter_proposals(regions, image_shape, min_area=400, max_aspect_ratio=4.0):
    height, width = image_shape[:2]
    proposals = []
    seen = set()
    for region in regions:
        x, y, w, h = region["rect"]
        if (x, y, w, h) in seen:
            continue
        if w <= 0 or h <= 0:
            continue
        if x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        if w * h < min_area:
            continue
        if max(w / h, h / w) > max_aspect_ratio:
            continue
        seen.add((x, y, w, h))
        proposals.append((int(x), int(y), int(w), int(h)))
    return proposals


def generate_or_load_proposals(split_name, split_items, cache_dir, args):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_proposals.json")
    if os.path.exists(cache_path) and not args.regenerate_proposals:
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    proposal_map = {}
    scales = parse_int_list(args.scales) or [int(args.scale)]
    for item in split_items:
        image = skimage.io.imread(item["path"])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        all_regions = []
        for scale in scales:
            _, regions = selective_search(
                image,
                scale=scale,
                sigma=args.sigma,
                min_size=args.min_size,
            )
            all_regions.extend(regions)
        proposals = filter_proposals(
            all_regions,
            image.shape,
            min_area=args.min_area,
            max_aspect_ratio=args.max_aspect_ratio,
        )
        proposal_map[item["file_name"]] = proposals[:args.max_proposals]
        print(
            f"{split_name}/{item['file_name']}: "
            f"{len(proposals)} proposals from scales {scales}"
        )

    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(proposal_map, handle)
    return proposal_map


def generate_proposals_for_scales(items, scales, args):
    proposal_map = {}
    for item in items:
        image = skimage.io.imread(item["path"])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        all_regions = []
        for scale in scales:
            _, regions = selective_search(
                image,
                scale=scale,
                sigma=args.sigma,
                min_size=args.min_size,
            )
            all_regions.extend(regions)

        proposals = filter_proposals(
            all_regions,
            image.shape,
            min_area=args.min_area,
            max_aspect_ratio=args.max_aspect_ratio,
        )
        proposal_map[item["file_name"]] = proposals[:args.max_proposals]
    return proposal_map


def print_mabo_scale_comparison(items, args, split_name):
    single_scale = [int(args.scale)]
    multi_scales = parse_int_list(args.scales) or single_scale
    single_proposals = generate_proposals_for_scales(items, single_scale, args)
    multi_proposals = generate_proposals_for_scales(items, multi_scales, args)
    print(
        f"{split_name} MABO single-scale {single_scale}: "
        f"{compute_mabo(items, single_proposals):.4f}"
    )
    print(
        f"{split_name} MABO multi-scale {multi_scales}: "
        f"{compute_mabo(items, multi_proposals):.4f}"
    )


def crop_box(image, box):
    x, y, w, h = [int(value) for value in box]
    return image[y:y + h, x:x + w]


def colour_shape_features(crop, original_shape):
    original_h, original_w = original_shape[:2]
    aspect = original_w / max(original_h, 1)
    clipped_aspect = min(aspect, 4.0) / 4.0
    squareness = min(original_w, original_h) / max(original_w, original_h, 1)
    area = float(original_w * original_h)
    perimeter = float(2 * (original_w + original_h))
    bbox_circularity = (4.0 * np.pi * area) / (perimeter ** 2 + 1e-6)

    hsv = skimage.color.rgb2hsv(crop)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated_mask = (saturation > 0.35) & (value > 0.20)
    saturated_fraction = float(saturated_mask.mean())

    labels = skimage.measure.label(saturated_mask)
    props = skimage.measure.regionprops(labels)
    if props:
        largest = max(props, key=lambda prop: prop.area)
        min_row, min_col, max_row, max_col = largest.bbox
        comp_h = max_row - min_row
        comp_w = max_col - min_col
        largest_fraction = float(largest.area) / float(crop.shape[0] * crop.shape[1])
        largest_extent = float(largest.extent)
        component_squareness = min(comp_w, comp_h) / max(comp_w, comp_h, 1)
    else:
        largest_fraction = 0.0
        largest_extent = 0.0
        component_squareness = 0.0

    gray = skimage.color.rgb2gray(crop)
    edge_density = float(skimage.feature.canny(gray, sigma=1.0).mean())

    red = crop[:, :, 0]
    green = crop[:, :, 1]
    blue = crop[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2))

    return np.array([
        clipped_aspect,
        squareness,
        bbox_circularity,
        float(saturation.mean()),
        float(saturation.std()),
        float(value.mean()),
        saturated_fraction,
        largest_fraction,
        largest_extent,
        component_squareness,
        edge_density + colorfulness,
    ], dtype=np.float32)


def balloon_prior_score(image, box):
    crop = crop_box(image, box)
    if crop.size == 0:
        return 0.0

    resized = skimage.transform.resize(
        crop, (64, 64), anti_aliasing=True, preserve_range=False
    )
    features = colour_shape_features(resized, crop.shape)
    squareness = features[1]
    saturated_fraction = features[6]
    largest_fraction = features[7]
    largest_extent = features[8]
    component_squareness = features[9]

    colour_term = min(saturated_fraction / 0.18, 1.0)
    component_term = min(largest_fraction / 0.16, 1.0)
    shape_term = 0.5 * squareness + 0.5 * component_squareness
    extent_term = min(largest_extent / 0.70, 1.0)
    return float(
        0.30 * colour_term
        + 0.30 * component_term
        + 0.30 * shape_term
        + 0.10 * extent_term
    )


def extract_features_from_crop(crop):
    if crop.size == 0:
        return np.zeros(FEATURE_LENGTH, dtype=np.float32)

    original_shape = crop.shape
    crop = skimage.transform.resize(
        crop, (64, 64), anti_aliasing=True, preserve_range=False
    )
    gray = skimage.color.rgb2gray(crop)
    hog = skimage.feature.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    hsv = skimage.color.rgb2hsv(crop)
    colour_hist = []
    for channel in range(3):
        hist, _ = np.histogram(hsv[:, :, channel], bins=16, range=(0.0, 1.0))
        colour_hist.extend(hist.astype(np.float32))
    colour_hist = np.array(colour_hist, dtype=np.float32)
    if colour_hist.sum() > 0:
        colour_hist /= colour_hist.sum()

    extra = colour_shape_features(crop, original_shape)
    return np.concatenate([hog.astype(np.float32), colour_hist, extra])


def extract_features(image, box):
    return extract_features_from_crop(crop_box(image, box))


def augment_crop(crop):
    variants = [crop]
    if crop.size == 0:
        return variants

    variants.append(crop[:, ::-1])
    crop_float = skimage.util.img_as_float32(crop)
    for delta in (-0.10, 0.10):
        jittered = np.clip(crop_float + delta, 0.0, 1.0)
        variants.append((jittered * 255).astype(np.uint8))
    return variants


def build_samples(
    items,
    proposal_map,
    pos_threshold,
    neg_threshold,
    max_negatives,
    max_hard_negatives,
    augment_positives=True,
    include_gt_positives=False,
):
    features = []
    labels = []

    for item in items:
        image = skimage.io.imread(item["path"])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        positives = []
        negatives = []
        hard_negatives = []
        gt_boxes = item["boxes"]
        for proposal in proposal_map.get(item["file_name"], []):
            best_iou = max([iou(proposal, gt) for gt in gt_boxes], default=0.0)
            if best_iou >= pos_threshold:
                positives.append(proposal)
            elif best_iou <= neg_threshold:
                negatives.append(proposal)
            elif best_iou < pos_threshold:
                hard_negatives.append(proposal)

        random.shuffle(negatives)
        random.shuffle(hard_negatives)
        negatives = negatives[:max_negatives]
        hard_negatives = hard_negatives[:max_hard_negatives]

        if include_gt_positives:
            positives.extend(gt_boxes)

        for proposal in positives:
            if augment_positives:
                for crop in augment_crop(crop_box(image, proposal)):
                    features.append(extract_features_from_crop(crop))
                    labels.append(1)
            else:
                features.append(extract_features(image, proposal))
                labels.append(1)
        for proposal in negatives:
            features.append(extract_features(image, proposal))
            labels.append(0)
        for proposal in hard_negatives:
            features.append(extract_features(image, proposal))
            labels.append(0)

    if not features:
        return np.empty((0, FEATURE_LENGTH), dtype=np.float32), np.array([])
    return np.array(features), np.array(labels)


def select_score_threshold(model, x_valid, y_valid):
    if len(y_valid) == 0 or len(np.unique(y_valid)) < 2:
        return 0.0

    scores = model.decision_function(x_valid)
    candidates = np.unique(np.percentile(scores, np.linspace(5, 95, 37)))
    best_threshold = 0.0
    best_f1 = -1.0
    best_precision = -1.0

    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_valid, pred, zero_division=0)
        precision = precision_score(y_valid, pred, zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and precision > best_precision):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_precision = float(precision)

    print(
        f"selected score threshold={best_threshold:.4f} "
        f"(validation F1={best_f1:.3f}, precision={best_precision:.3f})"
    )
    return best_threshold


def compute_mabo(items, proposal_map):
    best_overlaps = []
    for item in items:
        proposals = proposal_map.get(item["file_name"], [])
        for gt_box in item["boxes"]:
            if not proposals:
                best_overlaps.append(0.0)
            else:
                best_overlaps.append(max(iou(proposal, gt_box) for proposal in proposals))
    return float(np.mean(best_overlaps)) if best_overlaps else 0.0


def average_precision(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    changed = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changed + 1] - recalls[changed]) * precisions[changed + 1]))


def evaluate_predictions(items, predictions, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [round(value, 2) for value in np.arange(0.50, 1.00, 0.05)]

    gt_by_image = {item["id"]: list(item["boxes"]) for item in items}
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return {"AP50": 0.0, "mAP50_95": 0.0}

    predictions = sorted(predictions, key=lambda pred: pred["score"], reverse=True)
    aps = {}
    for threshold in iou_thresholds:
        matched = {image_id: set() for image_id in gt_by_image}
        tp = []
        fp = []
        for pred in predictions:
            image_id = pred["image_id"]
            gt_boxes = gt_by_image.get(image_id, [])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched[image_id]:
                    continue
                overlap = iou(pred["bbox"], gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_idx = gt_idx

            if best_iou >= threshold and best_gt_idx >= 0:
                matched[image_id].add(best_gt_idx)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        if not tp:
            aps[threshold] = 0.0
            continue

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recalls = tp / total_gt
        precisions = tp / np.maximum(tp + fp, 1e-8)
        aps[threshold] = average_precision(recalls, precisions)

    map_thresholds = [key for key in aps if 0.50 <= key <= 0.95]
    return {
        "AP50": aps.get(0.50, 0.0),
        "mAP50_95": float(np.mean([aps[key] for key in map_thresholds])),
    }


def save_prediction_image(image, boxes, scores, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for box, score in zip(boxes, scores):
        x, y, w, h = box
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, f"{score:.2f}", color="white",
                bbox={"facecolor": "red", "alpha": 0.6, "pad": 1})
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def predict_split(
    items,
    proposal_map,
    model,
    output_dir,
    max_detections,
    nms_threshold,
    score_threshold,
    min_balloon_prior,
):
    os.makedirs(output_dir, exist_ok=True)
    coco_predictions = []
    for item in items:
        image = skimage.io.imread(item["path"])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        proposals = proposal_map.get(item["file_name"], [])
        if not proposals:
            continue

        features = np.array([extract_features(image, proposal) for proposal in proposals])
        scores = model.decision_function(features)
        positive_indices = np.where(scores >= score_threshold)[0]

        boxes = []
        positive_scores = []
        for idx in positive_indices:
            prior = balloon_prior_score(image, proposals[idx])
            if prior < min_balloon_prior:
                continue
            boxes.append(proposals[idx])
            positive_scores.append(float(scores[idx] + 0.25 * prior))

        keep = nms(boxes, positive_scores, threshold=nms_threshold)
        keep = keep[:max_detections]

        final_boxes = [boxes[idx] for idx in keep]
        final_scores = [positive_scores[idx] for idx in keep]
        for box, score in zip(final_boxes, final_scores):
            coco_predictions.append({
                "image_id": item["id"],
                "category_id": 1,
                "bbox": [float(value) for value in box],
                "score": float(score),
            })
        output_path = os.path.join(output_dir, item["file_name"])
        save_prediction_image(image, final_boxes, final_scores, output_path)
        print(f"prediction {item['file_name']}: {len(final_boxes)} detections")
    return coco_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train a balloon SVM detector.")
    parser.add_argument("--data-dir", default="../data/balloon_dataset")
    parser.add_argument("--results-dir", default="../results/balloon_detector")
    parser.add_argument("--scale", type=float, default=150)
    parser.add_argument("--scales", default="100,150,250")
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--max-proposals", type=int, default=300)
    parser.add_argument("--pos-threshold", type=float, default=0.5)
    parser.add_argument("--neg-threshold", type=float, default=0.25)
    parser.add_argument("--max-negatives", type=int, default=80)
    parser.add_argument("--max-hard-negatives", type=int, default=40)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-balloon-prior", type=float, default=0.10)
    parser.add_argument("--svm-c-values", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--compare-mabo-scales", action="store_true")
    parser.add_argument("--no-gt-positives", action="store_true")
    parser.add_argument("--no-augment-positives", action="store_true")
    parser.add_argument("--regenerate-proposals", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(7)
    np.random.seed(7)

    train_items = load_coco(os.path.join(args.data_dir, "train"))
    valid_items = load_coco(os.path.join(args.data_dir, "valid"))
    test_items = load_coco(os.path.join(args.data_dir, "test"))

    cache_dir = os.path.join(args.results_dir, "proposal_cache")
    train_proposals = generate_or_load_proposals("train", train_items, cache_dir, args)
    valid_proposals = generate_or_load_proposals("valid", valid_items, cache_dir, args)
    test_proposals = generate_or_load_proposals("test", test_items, cache_dir, args)

    print(f"Train MABO: {compute_mabo(train_items, train_proposals):.4f}")
    print(f"Valid MABO: {compute_mabo(valid_items, valid_proposals):.4f}")
    print(f"Test MABO: {compute_mabo(test_items, test_proposals):.4f}")
    if args.compare_mabo_scales:
        print_mabo_scale_comparison(valid_items, args, "Valid")

    x_train, y_train = build_samples(
        train_items,
        train_proposals,
        args.pos_threshold,
        args.neg_threshold,
        args.max_negatives,
        args.max_hard_negatives,
        augment_positives=not args.no_augment_positives,
        include_gt_positives=not args.no_gt_positives,
    )
    x_valid, y_valid = build_samples(
        valid_items,
        valid_proposals,
        args.pos_threshold,
        args.neg_threshold,
        args.max_negatives,
        args.max_hard_negatives,
        augment_positives=False,
        include_gt_positives=False,
    )

    print(f"training samples: {len(y_train)}; positives: {int(y_train.sum())}")
    print(f"validation samples: {len(y_valid)}; positives: {int(y_valid.sum())}")
    print(f"training class ratio: {(y_train == 0).sum()} negative / {(y_train == 1).sum()} positive")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Need both positive and negative samples to train the SVM.")

    best_c = 1.0
    best_f1 = -1.0
    c_values = [float(value.strip()) for value in args.svm_c_values.split(",") if value.strip()]
    if len(y_valid) > 0 and len(np.unique(y_valid)) == 2:
        for c_value in c_values:
            candidate = make_pipeline(
                StandardScaler(),
                LinearSVC(C=c_value, class_weight="balanced", max_iter=10000),
            )
            candidate.fit(x_train, y_train)
            valid_pred = candidate.predict(x_valid)
            score = f1_score(y_valid, valid_pred, zero_division=0)
            print(f"C={c_value}: validation F1={score:.3f}")
            if score > best_f1:
                best_f1 = score
                best_c = c_value

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=best_c, class_weight="balanced", max_iter=10000),
    )
    model.fit(x_train, y_train)
    print(f"selected LinearSVC C={best_c}")
    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = select_score_threshold(model, x_valid, y_valid)
    else:
        print(f"using manual score threshold={score_threshold:.4f}")

    if len(y_valid) > 0:
        valid_scores = model.decision_function(x_valid)
        valid_pred = (valid_scores >= score_threshold).astype(int)
        print(classification_report(
            y_valid,
            valid_pred,
            target_names=["background", "balloon"],
            zero_division=0,
        ))

    os.makedirs(args.results_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.results_dir, "balloon_svm.joblib"))
    test_predictions = predict_split(
        test_items,
        test_proposals,
        model,
        os.path.join(args.results_dir, "test_predictions"),
        args.max_detections,
        args.nms_threshold,
        score_threshold,
        args.min_balloon_prior,
    )
    predictions_path = os.path.join(args.results_dir, "test_predictions_coco.json")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        json.dump(test_predictions, handle, indent=2)
    metrics = evaluate_predictions(test_items, test_predictions)
    print(f"Test AP@0.50: {metrics['AP50']:.4f}")
    print(f"Test mAP@0.50:0.95: {metrics['mAP50_95']:.4f}")
    metrics_path = os.path.join(args.results_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({
            "score_threshold": score_threshold,
            "min_balloon_prior": args.min_balloon_prior,
            "AP50": metrics["AP50"],
            "mAP50_95": metrics["mAP50_95"],
        }, handle, indent=2)


if __name__ == "__main__":
    main()
