import argparse

import cv2
import numpy as np

from cvproj_exc.config import Config, ReIdMode, enum_choices
from cvproj_exc.face_detector import FaceDetector
from cvproj_exc.face_recognition import FaceClustering, FaceRecognizer

# The test module of the face recognition system. This comprises the following workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and perform face identification (mode "ident") or re-identification
#      (mode "cluster").
#   4) Display face detection / tracking along with the prediction of face identification.


def main(args):
    # Setup OpenCV video capture.
    if args.video == "none":
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 200
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    # Image display
    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    # Prepare face detection, identification, and clustering.
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    clustering = FaceClustering()

    # The video capturing loop.
    while True:
        key = cv2.waitKey(wait_for_frame)

        # Stop capturing using ESC.
        if (key & 255) == 27:
            break

        # Pause capturing using 'p'.
        if key == ord("p"):
            cv2.waitKey(-1)

        # Capture new video frame.
        _, frame = camera.read()
        if frame is None:
            print("End of stream")
            break
        # Resize the frame.
        height, width = frame.shape[:2]
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s * width), int(s * height)))
        # Flip frame if it is live video.
        if args.video == "none":
            frame = cv2.flip(frame, 1)

        # Track (or initially detect if required) a face in the current frame.
        if (face := detector.track_face(frame)) is not None:
            if args.mode == ReIdMode.IDENT:
                # Face identification: predict identity for the current frame.
                predicted_label, prob, dist_to_prediction = recognizer.predict(face.aligned)
                label_str = f"{predicted_label}"
                confidence_str = f"Prob.: {prob:.2f}, Dist.: {dist_to_prediction:.2f}"
            elif args.mode == ReIdMode.CLUSTER:
                # Face clustering: determine cluster for the current frame.
                predicted_label, distances_to_clusters = clustering.predict(face.aligned)
                label_str = f"Cluster {predicted_label}"
                confidence_str = f"Dist.: {np.array2string(distances_to_clusters, precision=2)}"
            else:
                raise ValueError(f"Unknown prediction mode {args.mode}")

            state_str = f"{label_str} | {confidence_str}"
            face_rect = face.rect
            color = (0, 255, 0)
            if isinstance(predicted_label, str) and predicted_label.lower() == "unknown":
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (face_rect[0], face_rect[1]),
                (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1),
                color,
                2,
            )
            ((tw, th), _) = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (face_rect[0] - 1, face_rect[1] + face_rect[3]),
                (face_rect[0] + 1 + tw, face_rect[1] + face_rect[3] + th + 4),
                color,
                -1,
            )
            cv2.putText(
                frame,
                state_str,
                (face_rect[0], face_rect[1] + face_rect[3] + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Camera", frame)


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=ReIdMode,
        choices=enum_choices(ReIdMode),
        default=ReIdMode.IDENT,
        help="The test mode.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=Config.TEST_DATA.joinpath("Alan_Ball", "%04d.jpg"),
        help="The video capture input. In case of 'none' the default video capture (webcam) is "
        "used. Use a filename(s) to read video data from image file (see VideoCapture "
        "documentation).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(arguments())
