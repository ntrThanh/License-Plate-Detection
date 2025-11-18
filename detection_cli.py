import argparse
import cv2
from detection import DetectionLicensePlate

def get_args():
    parser = argparse.ArgumentParser(
        description='Program detects license plates'
    )

    parser.add_argument(
        '--image',
        '-i',
        type=str,
        help='Path to image file'
    )

    parser.add_argument(
        '--video',
        '-v',
        type=str,
        help='Path to video file'
    )

    parser.add_argument(
        '--camera',
        '-c',
        action='store_true',
        help='Use Camera'
    )

    parser.add_argument(
        '--checkpoint-detect',
        '-cdt',
        type=str,
        help='Path to YOLO detect checkpoint'
    )

    parser.add_argument(
        '--checkpoint-classify',
        '-cclf',
        type=str,
        help='Path to YOLO classify checkpoint'
    )

    return parser.parse_args()


def use_image(path_to_image, model):
    image = cv2.imread(path_to_image)

    if image is None:
        print(" Cannot read image.")
        return

    _, image_out = model.detect(image)

    cv2.imshow(
        "Result",
        image_out if image_out is not None else image
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def use_video(path_to_video, model):
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print("Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, image_out = model.detect(frame)

        cv2.imshow(
            "Video",
            image_out if image_out is not None else frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def use_camera(model, cam_id=0):
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, image_out = model.detect(frame)

        cv2.imshow(
            "Camera",
            image_out if image_out is not None else frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()

    if not args.checkpoint_detect and not args.checkpoint_classify:
        model = DetectionLicensePlate(
            checkpoint_detect='runs/detect/yolo detects license plate/weights/best.pt',
            checkpoint_classify='checkpoint/best_yolo_classified/best.pt'
        )
    else:
        model = DetectionLicensePlate(
            checkpoint_detect=args.checkpoint_detect,
            checkpoint_classify=args.checkpoint_classify
        )

    # priority image -> video -> camera
    if args.image:
        use_image(args.image, model)

    elif args.video:
        use_video(args.video, model)

    elif args.camera:
        use_camera(model)

    else:
        print(
            "Please provide input: "
            "--image / "
            "--video / "
            "--camera"
        )
