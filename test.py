import cv2

from detection import DetectionLicensePlate

if __name__ == '__main__':
    path_image = '/home/trong-thanh/Downloads/Dieu_0370.png'
    image = cv2.imread(path_image)

    model = DetectionLicensePlate(
        checkpoint_detect='runs/detect/yolo detects license plate/weights/best.pt',
        checkpoint_classify='checkpoint/best_yolo_classified/best.pt'
    )

    image_out = model.detect(image)[1]

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    