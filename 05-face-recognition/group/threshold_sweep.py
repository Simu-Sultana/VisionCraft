import csv
import pickle
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from cvproj_exc.config import Config
from cvproj_exc.face_detector import FaceDetector
from cvproj_exc.face_recognition import FaceNet

TEST_DIR = Path("data/test_data")
OUT_LOG = Path("results/logs/threshold_sweep.csv")
OUT_SUMMARY = Path("results/logs/threshold_sweep_summary.txt")

KNOWN_LABELS = {
    "Alan_Ball",
    "Manuel_Pellegrini",
    "Marina_Silva",
    "Nancy_Sinatra",
    "Peter_Gilmour",
}

with open(Config.REC_GALLERY, "rb") as f:
    gallery_labels, gallery_embeddings = pickle.load(f)

gallery_labels = np.array(gallery_labels)
gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)

facenet = FaceNet()

def two_embeddings(face):
    emb_color = facenet.predict(face)

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    emb_gray = facenet.predict(gray_bgr)

    return np.stack([emb_color, emb_gray], axis=0)

def raw_knn_predict(face, k=3):
    query_embeddings = two_embeddings(face)

    best_dist_per_gallery = []
    for ge in gallery_embeddings:
        d = np.linalg.norm(query_embeddings - ge[None, :], axis=1)
        best_dist_per_gallery.append(np.min(d))

    distances = np.asarray(best_dist_per_gallery)
    nn_idx = np.argsort(distances)[:k]
    nn_labels = gallery_labels[nn_idx]
    nn_dists = distances[nn_idx]

    counts = Counter(nn_labels)
    pred_label, pred_count = counts.most_common(1)[0]
    prob = pred_count / k
    pred_dist = np.min(nn_dists[nn_labels == pred_label])

    return str(pred_label), float(prob), float(pred_dist)

rows = []

print("Extracting raw predictions from test set...")

for person_dir in sorted(TEST_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    true_name = person_dir.name
    image_files = sorted(person_dir.glob("*.jpg"))

    detector = FaceDetector()

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue

        face = detector.track_face(frame)
        if face is None:
            continue

        raw_label, prob, dist = raw_knn_predict(face.aligned, k=3)

        true_type = "known" if true_name in KNOWN_LABELS else "unknown"
        rows.append([true_name, true_type, image_file.name, raw_label, prob, dist])

OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true_folder", "true_type", "frame", "raw_pred_label", "probability", "distance"])
    writer.writerows(rows)

thresholds = np.arange(0.80, 1.51, 0.01)

summary_lines = []
summary_lines.append("threshold,known_accuracy,unknown_rejection,overall_score,known_correct,known_total,unknown_rejected,unknown_total")

best = None

for th in thresholds:
    known_correct = 0
    known_total = 0
    unknown_rejected = 0
    unknown_total = 0

    for true_name, true_type, frame, raw_label, prob, dist in rows:
        dist = float(dist)

        if dist > th or float(prob) < 0.34:
            final_label = "unknown"
        else:
            final_label = raw_label

        if true_type == "known":
            known_total += 1
            if final_label == true_name:
                known_correct += 1
        else:
            unknown_total += 1
            if final_label == "unknown":
                unknown_rejected += 1

    known_acc = known_correct / known_total if known_total else 0
    unk_rej = unknown_rejected / unknown_total if unknown_total else 0

    # Balanced score: known recognition and unknown rejection are equally important.
    overall = 0.5 * known_acc + 0.5 * unk_rej

    line = f"{th:.2f},{known_acc:.4f},{unk_rej:.4f},{overall:.4f},{known_correct},{known_total},{unknown_rejected},{unknown_total}"
    summary_lines.append(line)

    if best is None or overall > best[0]:
        best = (overall, th, known_acc, unk_rej, known_correct, known_total, unknown_rejected, unknown_total)

with open(OUT_SUMMARY, "w") as f:
    f.write("\n".join(summary_lines))
    f.write("\n\n")
    f.write("BEST THRESHOLD BY BALANCED SCORE\n")
    f.write(f"threshold: {best[1]:.2f}\n")
    f.write(f"known_accuracy: {best[2]:.4f} ({best[4]}/{best[5]})\n")
    f.write(f"unknown_rejection: {best[3]:.4f} ({best[6]}/{best[7]})\n")
    f.write(f"balanced_score: {best[0]:.4f}\n")

print(open(OUT_SUMMARY).read())
print("Saved:", OUT_LOG)
print("Saved:", OUT_SUMMARY)
