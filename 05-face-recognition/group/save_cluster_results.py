import csv
from pathlib import Path

import cv2

from cvproj_exc.face_detector import FaceDetector
from cvproj_exc.face_recognition import FaceClustering

TEST_DIR = Path("data/test_data")
OUT_IMG_DIR = Path("results/screenshots_cluster")
OUT_LOG = Path("results/logs/cluster_predictions.csv")

# We trained clustering with two known people only.
TEST_PEOPLE = ["Alan_Ball", "Manuel_Pellegrini"]

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

clusterer = FaceClustering()
rows = []

for person_name in TEST_PEOPLE:
    person_dir = TEST_DIR / person_name

    if not person_dir.exists():
        print(f"[SKIP] Missing test folder: {person_dir}")
        continue

    image_files = sorted(person_dir.glob("*.jpg"))
    detector = FaceDetector()

    saved_count = 0
    total_faces = 0

    print(f"\nTesting clustering for {person_name} with {len(image_files)} frames")

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue

        face = detector.track_face(frame)
        if face is None:
            rows.append([person_name, image_file.name, "NO_FACE", ""])
            continue

        total_faces += 1

        cluster_id, distances = clusterer.predict(face.aligned)

        distances_str = ";".join([f"{float(d):.4f}" for d in distances])
        rows.append([person_name, image_file.name, int(cluster_id), distances_str])

        x, y, w, h = face.rect
        text = f"Cluster {cluster_id} | dists: {distances_str}"

        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y + h), (x + 650, y + h + 28), (0, 255, 0), -1)
        cv2.putText(
            frame,
            text,
            (x + 4, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            1,
        )

        if saved_count < 5:
            out_file = OUT_IMG_DIR / f"cluster_{person_name}_{saved_count + 1}.png"
            cv2.imwrite(str(out_file), frame)
            saved_count += 1

    print(f"{person_name}:")
    print(f"  Faces detected   : {total_faces}")
    print(f"  Saved screenshots: {saved_count}")

with open(OUT_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true_folder", "frame", "predicted_cluster", "distance_distribution"])
    writer.writerows(rows)

print("\nSaved clustering screenshots to:", OUT_IMG_DIR)
print("Saved clustering CSV to:", OUT_LOG)
