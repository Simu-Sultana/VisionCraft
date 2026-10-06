import csv
from pathlib import Path

import cv2

from cvproj_exc.face_detector import FaceDetector
from cvproj_exc.face_recognition import FaceRecognizer

TEST_DIR = Path("data/test_data")
OUT_IMG_DIR = Path("results/screenshots")
OUT_LOG = Path("results/logs/ident_predictions.csv")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

recognizer = FaceRecognizer(num_neighbours=3, max_distance=1.02, min_prob=0.34)

rows = []

for person_dir in sorted(TEST_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    person_name = person_dir.name
    image_files = sorted(person_dir.glob("*.jpg"))

    if not image_files:
        print(f"[SKIP] No jpg files found in {person_dir}")
        continue

    detector = FaceDetector()

    saved_count = 0
    total_faces = 0
    correct_count = 0
    unknown_count = 0

    print(f"\nTesting {person_name} with {len(image_files)} frames")

    for frame_idx, image_file in enumerate(image_files):
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue

        face = detector.track_face(frame)

        if face is None:
            rows.append([person_name, image_file.name, "NO_FACE", "", ""])
            continue

        total_faces += 1

        predicted_label, prob, dist = recognizer.predict(face.aligned)

        if predicted_label == person_name:
            correct_count += 1

        if isinstance(predicted_label, str) and predicted_label.lower() == "unknown":
            unknown_count += 1

        rows.append([person_name, image_file.name, predicted_label, f"{prob:.4f}", f"{dist:.4f}"])

        # Draw annotation
        x, y, w, h = face.rect
        color = (0, 255, 0)
        if isinstance(predicted_label, str) and predicted_label.lower() == "unknown":
            color = (0, 0, 255)

        text = f"{predicted_label} | Prob.: {prob:.2f}, Dist.: {dist:.2f}"

        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.rectangle(frame, (x, y + h), (x + 520, y + h + 28), color, -1)
        cv2.putText(frame, text, (x + 4, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Save only a few proof images per person
        if saved_count < 5:
            out_file = OUT_IMG_DIR / f"ident_{person_name}_{saved_count + 1}.png"
            cv2.imwrite(str(out_file), frame)
            saved_count += 1

    accuracy = correct_count / total_faces if total_faces > 0 else 0.0

    print(f"{person_name}:")
    print(f"  Faces detected   : {total_faces}")
    print(f"  Correct frames   : {correct_count}")
    print(f"  Unknown frames   : {unknown_count}")
    print(f"  Visual accuracy  : {accuracy:.4f}")
    print(f"  Saved screenshots: {saved_count}")

# Save CSV
with open(OUT_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true_folder", "frame", "predicted_label", "probability", "distance"])
    writer.writerows(rows)

print("\nSaved identification screenshots to:", OUT_IMG_DIR)
print("Saved prediction CSV to:", OUT_LOG)
